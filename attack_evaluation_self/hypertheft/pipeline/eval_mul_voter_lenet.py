import argparse
import csv
import json
import math
import random
from pathlib import Path

import importlib
import numpy as np
import torch

from dataloader_trace_bridge import load_task_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one HyperTheft checkpoint with mul-voter.")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--mode", default="off", choices=["off", "on"], type=str)
    parser.add_argument("--target_model", default=None, choices=["lenet64", "conv32"], type=str)
    parser.add_argument("--num_vote", default="1,5,11,21", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_run_args(run_dir):
    config = json.load(open(run_dir / "config.json", "r", encoding="utf-8"))
    args = config["args"]

    class Args:
        pass

    cfg = Args()
    for k, v in args.items():
        setattr(cfg, k, v)
    return cfg


def load_checkpoint(ckpt_path, hypertheft):
    state = torch.load(ckpt_path, map_location="cpu")
    hypertheft.mixer.load_state_dict(state["mixer"], strict=True)
    for g_idx, gen in enumerate(hypertheft.generator.as_list()):
        gen.load_state_dict(state[f"G{g_idx}"], strict=True)


def load_heldout_task(cfg, mode, task_name):
    tasks = load_task_loaders(
        trace_root=cfg.trace_root,
        mode=mode,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        fold_trace=cfg.fold_trace,
        trace_c=cfg.trace_c,
        trace_w=cfg.trace_w,
        trace_len=cfg.trace_len,
        image_size=cfg.image_size,
        granularity=cfg.granularity,
        include_heldout=True,
    )
    matched = [task for task in tasks if task["task_name"] == task_name]
    if len(matched) != 1:
        names = [task["task_name"] for task in tasks]
        raise RuntimeError(
            f"Expected exactly one task named {task_name}, found {len(matched)}. "
            f"Available tasks: {names}"
        )
    return matched[0]


def build_codes(cfg, hypertheft, task, device):
    codes_list = []
    for seed_id, (inter_out, _, _) in enumerate(task["train_loader"]):
        if seed_id >= cfg.n_seed:
            break
        inter_out = inter_out[: cfg.batch_size].to(device)
        codes_list.append(hypertheft.mixer(inter_out))
    if len(codes_list) != cfg.n_seed:
        raise RuntimeError(f"Only collected {len(codes_list)} seed batches, expected {cfg.n_seed}")
    return codes_list


@torch.no_grad()
def evaluate_votes(cfg, hypertheft, task, codes_list, num_vote_list, ckpt_name, epoch):
    results = []
    device = cfg.device
    for num_vote in num_vote_list:
        params_list = []
        for _ in range(num_vote):
            params_list.append(hypertheft.generator(torch.cat(codes_list, -1)))

        correct = torch.zeros(cfg.batch_size, device=device)
        total = torch.zeros(cfg.batch_size, device=device)
        for _, images, labels in task["eval_loader"]:
            images = images.to(device)
            labels = labels.to(device)
            counter = torch.zeros(cfg.batch_size, labels.size(0), cfg.n_class, device=device)
            for params in params_list:
                batch_id = 0
                for layers in zip(*params):
                    out = hypertheft.eval_f(layers, images)
                    pred = out.argmax(-1)
                    counter[batch_id, torch.arange(pred.size(0), device=device), pred] += 1
                    batch_id += 1
            voted_pred = counter.argmax(-1)
            voted_correct = voted_pred == labels.repeat(cfg.batch_size, 1)
            mask = (counter > math.floor(num_vote * 0.75)).int().sum(-1)
            correct += (voted_correct * mask).sum(-1)
            total += (torch.ones_like(voted_correct) * mask).sum(-1)

        acc = torch.where(total > 0, correct / total.clamp_min(1), torch.zeros_like(total))
        results.append(
            {
                "checkpoint": ckpt_name,
                "epoch": epoch,
                "task_name": task["task_name"],
                "num_vote": num_vote,
                "max_acc": float(acc.max().item()),
                "avg_acc": float(acc.mean().item()),
                "min_acc": float(acc.min().item()),
                "nonzero_models": int((total > 0).sum().item()),
                "batch_size": int(cfg.batch_size),
            }
        )
    return results


def infer_epoch(ckpt_path):
    stem = ckpt_path.stem
    if stem.isdigit():
        return int(stem)
    return -1


def save_results(output_dir, stem, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{stem}.json"
    out_csv = output_dir / f"{stem}.csv"

    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "epoch",
                "task_name",
                "num_vote",
                "max_acc",
                "avg_acc",
                "min_acc",
                "nonzero_models",
                "batch_size",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(results, indent=2))
    print(out_json)
    print(out_csv)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "mul_voter_eval"

    cfg = load_run_args(run_dir)
    cfg.device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if args.target_model is not None:
        cfg.target_model = args.target_model

    set_seed(args.seed)

    models = importlib.import_module(f"models.{cfg.target_model}")
    hypertheft = models.HyperTheft(cfg)
    hypertheft.mixer.eval()
    for gen in hypertheft.generator.as_list():
        gen.eval()

    load_checkpoint(ckpt_path, hypertheft)
    task = load_heldout_task(cfg, args.mode, args.task_name)
    codes_list = build_codes(cfg, hypertheft, task, cfg.device)

    num_vote_list = [int(item) for item in args.num_vote.split(",") if item.strip()]
    results = evaluate_votes(
        cfg=cfg,
        hypertheft=hypertheft,
        task=task,
        codes_list=codes_list,
        num_vote_list=num_vote_list,
        ckpt_name=ckpt_path.name,
        epoch=infer_epoch(ckpt_path),
    )
    stem = f"{cfg.target_model}_{args.mode}_{args.task_name}_{ckpt_path.stem}_mul_voter"
    save_results(output_dir, stem, results)


if __name__ == "__main__":
    main()
