import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import importlib

import torch
import torch.nn.functional as F

from dataloader_trace_bridge import inspect_trace_layout, load_task_loaders
def parse_args():
    parser = argparse.ArgumentParser(description="Train HyperTheft on binary-task traces")
    parser.add_argument("--trace_root", required=True, type=str)
    parser.add_argument("--mode", required=True, choices=["off", "on"])
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--task_limit", default=None, type=int)
    parser.add_argument(
        "--task_names",
        default=None,
        type=str,
        help="Comma-separated task directory names to include (e.g., task00_0vs1,task01_0vs2).",
    )
    parser.add_argument("--eval_every", default=5, type=int)
    parser.add_argument("--save_every", default=5, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--z", default=64, type=int)
    parser.add_argument("--n_seed", default=1, type=int)
    parser.add_argument("--noise_weight", default=1.0, type=float)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--fold_trace", default=1, type=int, choices=[0, 1])
    parser.add_argument("--granularity", default=1, type=int)
    parser.add_argument("--trace_c", default=None, type=int)
    parser.add_argument("--trace_w", default=None, type=int)
    parser.add_argument("--trace_len", default=None, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--n_ch", default=1, type=int)
    parser.add_argument("--num_voters", default="1", type=str)
    parser.add_argument("--runtime", default="glow", type=str)
    parser.add_argument("--dataset", default="MNIST64", type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--target_model", default="lenet64", choices=["lenet64", "conv32"], type=str)
    parser.add_argument("--seed_cache_limit", default=0, type=int)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_runtime_args(args, trace_layout):
    args.trace_len = args.trace_len or trace_layout["trace_len"]
    args.trace_w = args.trace_w or trace_layout["trace_w"]
    args.trace_c = args.trace_c or trace_layout["trace_c"]
    args.ngen = args.batch_size
    return args


def collect_seed_traces(seed_loader, limit=None):
    chunks = []
    total = 0
    for traces, _, _ in seed_loader:
        if limit is not None and total >= limit:
            break
        if limit is not None:
            remain = max(limit - total, 0)
            if remain == 0:
                break
            traces = traces[:remain]
        chunks.append(traces)
        total += traces.size(0)
    if not chunks:
        raise RuntimeError("No seed traces were collected")
    return torch.cat(chunks, 0)


def prepare_task_caches(tasks, seed_cache_limit, batch_size, n_seed):
    base_limit = int(seed_cache_limit)
    if base_limit <= 0:
        base_limit = 64 if int(batch_size) <= 64 else 128
    target_limit = max(base_limit, int(batch_size) * int(n_seed))
    for task in tasks:
        if "seed_traces_cpu" not in task:
            task["seed_traces_cpu"] = collect_seed_traces(
                task["seed_loader"], limit=target_limit
            )
        if "eval_seed_traces_cpu" not in task:
            task["eval_seed_traces_cpu"] = collect_seed_traces(
                task["eval_seed_loader"], limit=target_limit
            )


def train_one_epoch(args, hypertheft, tasks, optimizers):
    mixer = hypertheft.mixer
    generator = hypertheft.generator
    optim_q, optim_g_list = optimizers
    mixer.train()
    for gen in generator.as_list():
        gen.train()

    task_summaries = []
    random.shuffle(tasks)
    for task in tasks:
        all_seed_traces = task["seed_traces_cpu"]
        total_seed = all_seed_traces.size(0)
        task_loss_sum = 0.0
        task_correct = 0.0
        task_total = 0.0

        for _, (traces, images, labels) in enumerate(task["train_loader"]):
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            perm = all_seed_traces[torch.randperm(total_seed)]
            inter_out_list = []
            for seed_id in range(args.n_seed):
                start = seed_id * args.batch_size
                end = start + args.batch_size
                inter_out = perm[start:end].to(args.device, non_blocking=True)
                inter_out_list.append(inter_out)

            codes_list = [mixer(inter_out) for inter_out in inter_out_list]
            params = generator(torch.cat(codes_list, -1))

            clf_loss = 0.0
            for layers in zip(*params):
                out = hypertheft.eval_f(layers, images)
                loss = F.cross_entropy(out, labels)
                pred = out.argmax(1)
                task_correct += pred.eq(labels).float().sum().item()
                task_total += labels.size(0)
                clf_loss += loss

            optim_q.zero_grad(set_to_none=True)
            for optim_g in optim_g_list:
                optim_g.zero_grad(set_to_none=True)

            total_loss = clf_loss / images.size(0)
            total_loss.backward()
            optim_q.step()
            for optim_g in optim_g_list:
                optim_g.step()
            task_loss_sum += float(total_loss.item())

        task_summaries.append(
            {
                "task_name": task["task_name"],
                "train_acc": task_correct / max(task_total, 1.0),
                "train_loss": task_loss_sum / max(len(task["train_loader"]), 1),
            }
        )
    return task_summaries


@torch.no_grad()
def evaluate(args, hypertheft, tasks, num_voters):
    mixer = hypertheft.mixer
    generator = hypertheft.generator
    mixer.eval()
    for gen in generator.as_list():
        gen.eval()

    task_summaries = []
    for task in tasks:
        eval_seed_traces = task["eval_seed_traces_cpu"]
        needed = args.batch_size * args.n_seed
        if eval_seed_traces.size(0) < needed:
            raise RuntimeError(
                f"Task {task['task_name']} only has {eval_seed_traces.size(0)} eval seeds, "
                f"but {needed} are required"
            )
        seed_batches = []
        for seed_index in range(args.n_seed):
            start = seed_index * args.batch_size
            end = start + args.batch_size
            seed_batches.append(
                eval_seed_traces[start:end].to(args.device, non_blocking=True)
            )
        codes_list = [mixer(trace_batch) for trace_batch in seed_batches]

        task_metrics = {"task_name": task["task_name"], "voters": []}
        for voter_count in num_voters:
            params_list = [generator(torch.cat(codes_list, -1)) for _ in range(voter_count)]
            correct = torch.zeros(args.batch_size, device=args.device)
            total = torch.zeros(args.batch_size, device=args.device)
            for _, images, labels in task["eval_loader"]:
                images = images.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                counter = torch.zeros(
                    args.batch_size, labels.size(0), args.n_class, device=args.device
                )
                for params in params_list:
                    batch_id = 0
                    for layers in zip(*params):
                        out = hypertheft.eval_f(layers, images)
                        pred = out.argmax(-1)
                        counter[batch_id, torch.arange(labels.size(0), device=args.device), pred] += 1
                        batch_id += 1
                voted_pred = counter.argmax(-1)
                voted_correct = voted_pred.eq(labels.repeat(args.batch_size, 1))
                threshold = math.floor(voter_count * 0.75)
                mask = (counter > threshold).int().sum(-1)
                correct += (voted_correct * mask).sum(-1)
                total += mask.sum(-1)
            acc = torch.where(total > 0, correct / total.clamp_min(1), torch.zeros_like(total))
            task_metrics["voters"].append(
                {
                    "num_voters": voter_count,
                    "max_acc": float(acc.max().item()),
                    "avg_acc": float(acc.mean().item()),
                }
            )
        task_summaries.append(task_metrics)
    return task_summaries


def save_checkpoint(output_dir, epoch, hypertheft, optim_q, optim_g_list, schedulers, history):
    state = {
        "epoch": epoch,
        "mixer": hypertheft.mixer.state_dict(),
        "Dz": hypertheft.discriminator.state_dict(),
        "optim_q": optim_q.state_dict(),
        "optim_g_list": [optim.state_dict() for optim in optim_g_list],
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
        "history": history,
    }
    for g_idx, gen in enumerate(hypertheft.generator.as_list()):
        state[f"G{g_idx}"] = gen.state_dict()
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / f"{epoch:03d}.pt")


def load_checkpoint(resume_path, hypertheft, optim_q, optim_g_list, schedulers):
    state = torch.load(resume_path, map_location="cpu")
    hypertheft.mixer.load_state_dict(state["mixer"], strict=True)
    hypertheft.discriminator.load_state_dict(state["Dz"], strict=True)
    for g_idx, gen in enumerate(hypertheft.generator.as_list()):
        gen.load_state_dict(state[f"G{g_idx}"], strict=True)
    if "optim_q" in state:
        optim_q.load_state_dict(state["optim_q"])
    if "optim_g_list" in state:
        for optim, optim_state in zip(optim_g_list, state["optim_g_list"]):
            optim.load_state_dict(optim_state)
    if "schedulers" in state:
        for scheduler, scheduler_state in zip(schedulers, state["schedulers"]):
            scheduler.load_state_dict(scheduler_state)
    start_epoch = int(state.get("epoch", -1)) + 1
    history = state.get("history", [])
    return start_epoch, history


def main():
    args = parse_args()
    include_task_names = None
    if args.task_names:
        include_task_names = [name.strip() for name in args.task_names.split(",") if name.strip()]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_layout = inspect_trace_layout(args.trace_root, args.mode, trace_w_override=args.trace_w)
    args = build_runtime_args(args, trace_layout)
    num_voters = [int(x) for x in args.num_voters.split(",") if x.strip()]

    config = {
        "args": vars(args),
        "trace_layout": trace_layout,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    models = importlib.import_module(f"models.{args.target_model}")
    hypertheft = models.HyperTheft(args)
    optim_q = torch.optim.Adam(
        hypertheft.mixer.parameters(), lr=args.lr, weight_decay=args.wd
    )
    optim_g_list = [
        torch.optim.Adam(gen.parameters(), lr=args.lr, weight_decay=args.wd)
        for gen in hypertheft.generator.as_list()
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optim_q, T_max=max(args.epochs, 1))
    ]
    schedulers.extend(
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 1))
        for optim in optim_g_list
    )

    start_epoch = 0
    history = []
    if args.resume:
        start_epoch, history = load_checkpoint(
            args.resume,
            hypertheft=hypertheft,
            optim_q=optim_q,
            optim_g_list=optim_g_list,
            schedulers=schedulers,
        )

    tasks = load_task_loaders(
        trace_root=args.trace_root,
        mode=args.mode,
        batch_size=args.batch_size,
        workers=args.workers,
        fold_trace=args.fold_trace,
        trace_c=args.trace_c,
        trace_w=args.trace_w,
        trace_len=args.trace_len,
        image_size=args.image_size,
        granularity=args.granularity,
        task_limit=args.task_limit,
        include_task_names=include_task_names,
    )
    prepare_task_caches(
        tasks,
        seed_cache_limit=args.seed_cache_limit,
        batch_size=args.batch_size,
        n_seed=args.n_seed,
    )

    for epoch in range(start_epoch, args.epochs):
        train_summary = train_one_epoch(
            args=args,
            hypertheft=hypertheft,
            tasks=tasks,
            optimizers=(optim_q, optim_g_list),
        )
        eval_summary = None
        if (epoch + 1) % args.eval_every == 0:
            eval_summary = evaluate(
                args=args,
                hypertheft=hypertheft,
                tasks=tasks,
                num_voters=num_voters,
            )
        for scheduler in schedulers:
            scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "train": train_summary,
            "eval": eval_summary,
        }
        history.append(epoch_record)
        with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir,
                epoch,
                hypertheft,
                optim_q=optim_q,
                optim_g_list=optim_g_list,
                schedulers=schedulers,
                history=history,
            )
        train_accs = [item["train_acc"] for item in train_summary]
        train_losses = [item["train_loss"] for item in train_summary]
        print(
            f"epoch={epoch} train_acc_mean={np.mean(train_accs):.4f} "
            f"train_loss_mean={np.mean(train_losses):.4f}"
        )
        if eval_summary is not None:
            eval_accs = [item["voters"][0]["avg_acc"] for item in eval_summary]
            print(f"epoch={epoch} eval_avg_acc_mean={np.mean(eval_accs):.4f}")


if __name__ == "__main__":
    main()
