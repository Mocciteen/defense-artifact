#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from dataloader_trace_bridge import inspect_trace_layout, load_task_loaders


DEFAULT_TRACE_FILES = ",".join(
    [
        "semantic_relu_relu04_layer2_block0_out_bits.txt",
        "semantic_relu_relu05_layer3_block0_conv1_bits.txt",
        "semantic_relu_relu06_layer3_block0_out_bits.txt",
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HyperTheft with strict paper-style same-class seed batches."
    )
    parser.add_argument("--trace_root", required=True, type=str)
    parser.add_argument("--mode", required=True, choices=["off", "on"])
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--trace_files",
        default=DEFAULT_TRACE_FILES,
        type=str,
        help="Comma-separated trace component files to concatenate per sample.",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--task_limit", default=None, type=int)
    parser.add_argument(
        "--task_names",
        default=None,
        type=str,
        help="Comma-separated task directory names to include.",
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
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--n_ch", default=3, type=int)
    parser.add_argument("--num_voters", default="1", type=str)
    parser.add_argument("--runtime", default="glow", type=str)
    parser.add_argument("--dataset", default="CIFAR10", type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--target_model",
        default="resnet32",
        choices=["resnet32", "resnet"],
        type=str,
    )
    parser.add_argument("--seed_cache_limit", default=0, type=int)
    parser.add_argument("--sanity_only", action="store_true")
    parser.add_argument("--sanity_batches", default=1, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp_dtype", default="bfloat16", choices=["float16", "bfloat16"]
    )
    return parser.parse_args()


def parse_trace_files_arg(raw: str):
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise RuntimeError("trace_files resolved to an empty list")
    return values


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_runtime_args(args, trace_layout):
    args.trace_len = args.trace_len or trace_layout["trace_len"]
    args.trace_w = args.trace_w or trace_layout["trace_w"]
    args.trace_c = args.trace_c or trace_layout["trace_c"]
    input_shape = trace_layout.get(
        "input_shape", [int(args.n_ch), int(args.image_size), int(args.image_size)]
    )
    args.input_shape = [int(value) for value in input_shape]
    args.n_ch = int(args.input_shape[0])
    args.image_size = int(args.input_shape[-1])
    return args


def is_cuda_device(device):
    return torch.device(device).type == "cuda"


def infer_ngen(model_module, args):
    if hasattr(model_module, "infer_ngen"):
        return int(model_module.infer_ngen(args))
    raise RuntimeError(
        f"Model module models.{args.target_model} does not expose infer_ngen(args)"
    )


def collect_seed_cache_by_label(seed_loader, per_label_limit=None):
    traces_by_label = {0: [], 1: []}
    counts = {0: 0, 1: 0}
    for traces, _, labels in seed_loader:
        for label_value in (0, 1):
            mask = labels == int(label_value)
            if not mask.any():
                continue
            label_traces = traces[mask]
            if per_label_limit is not None:
                if counts[label_value] >= per_label_limit:
                    continue
                remain = max(per_label_limit - counts[label_value], 0)
                if remain == 0:
                    continue
                label_traces = label_traces[:remain]
            if label_traces.numel() == 0:
                continue
            traces_by_label[label_value].append(label_traces)
            counts[label_value] += label_traces.size(0)
        if per_label_limit is not None and all(
            counts[label_value] >= per_label_limit for label_value in (0, 1)
        ):
            break
    out = {}
    for label_value in (0, 1):
        if not traces_by_label[label_value]:
            raise RuntimeError(
                f"No seed traces were collected for binary label {label_value}"
            )
        out[label_value] = torch.cat(traces_by_label[label_value], 0)
    return out


def sample_sameclass_seed_batches(
    seed_cache_by_label, batch_size, n_seed, source_label, device
):
    source_label = int(source_label)
    seed_batches = []
    for _ in range(int(n_seed)):
        pool = seed_cache_by_label[source_label]
        picks = torch.randint(pool.size(0), (int(batch_size),))
        seed_batches.append(pool[picks].to(device, non_blocking=True))
    return seed_batches


def build_paperflip_targets(labels, source_label):
    # Strict paper semantics for one training/eval step:
    # every seed trace in the step belongs to the same source class s, and the
    # generated surrogate answers YES iff the input also belongs to s.
    return labels.eq(int(source_label)).long()


def get_autocast_kwargs(args):
    enabled = bool(args.amp and is_cuda_device(args.device))
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    return {"device_type": "cuda", "enabled": enabled, "dtype": amp_dtype}


def prepare_task_caches(tasks, seed_cache_limit, batch_size, n_seed):
    base_limit = int(seed_cache_limit)
    if base_limit <= 0:
        base_limit = 64 if int(batch_size) <= 64 else 128
    target_limit = max(base_limit, int(batch_size) * int(n_seed))
    for task in tasks:
        if "seed_cache_by_label_cpu" not in task:
            task["seed_cache_by_label_cpu"] = collect_seed_cache_by_label(
                task["seed_loader"], per_label_limit=target_limit
            )
        if "eval_seed_cache_by_label_cpu" not in task:
            task["eval_seed_cache_by_label_cpu"] = collect_seed_cache_by_label(
                task["eval_seed_loader"], per_label_limit=target_limit
            )


def train_one_epoch(args, hypertheft, tasks, optimizers, scaler):
    mixer = hypertheft.mixer
    generator = hypertheft.generator
    optim_q, optim_g_list = optimizers
    mixer.train()
    for gen in generator.as_list():
        gen.train()

    task_summaries = []
    random.shuffle(tasks)
    for task in tasks:
        task_loss_sum = 0.0
        task_correct = 0.0
        task_total = 0.0
        start_label = random.randint(0, 1)

        for batch_index, (traces, images, labels) in enumerate(task["train_loader"]):
            del traces
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            source_label = int((start_label + batch_index) % 2)

            inter_out_list = sample_sameclass_seed_batches(
                task["seed_cache_by_label_cpu"],
                batch_size=args.batch_size,
                n_seed=args.n_seed,
                source_label=source_label,
                device=args.device,
            )
            targets = build_paperflip_targets(labels, source_label)

            with autocast(**get_autocast_kwargs(args)):
                codes_list = [mixer(inter_out) for inter_out in inter_out_list]
                params = generator(torch.cat(codes_list, -1))
                clf_loss = 0.0
                for layers in zip(*params):
                    out = hypertheft.eval_f(layers, images)
                    loss = F.cross_entropy(out, targets)
                    pred = out.argmax(1)
                    task_correct += pred.eq(targets).float().sum().item()
                    task_total += targets.size(0)
                    clf_loss += loss

            optim_q.zero_grad(set_to_none=True)
            for optim_g in optim_g_list:
                optim_g.zero_grad(set_to_none=True)

            total_loss = clf_loss / max(args.batch_size, 1)
            scaler.scale(total_loss).backward()
            scaler.step(optim_q)
            for optim_g in optim_g_list:
                scaler.step(optim_g)
            scaler.update()

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
        task_metrics = {"task_name": task["task_name"], "voters": []}
        for voter_count in num_voters:
            all_acc = []
            label_breakdown = []
            for source_label in (0, 1):
                seed_batches = sample_sameclass_seed_batches(
                    task["eval_seed_cache_by_label_cpu"],
                    batch_size=args.batch_size,
                    n_seed=args.n_seed,
                    source_label=source_label,
                    device=args.device,
                )
                codes_list = [mixer(trace_batch) for trace_batch in seed_batches]
                params_list = []
                for _ in range(voter_count):
                    with autocast(**get_autocast_kwargs(args)):
                        params_list.append(generator(torch.cat(codes_list, -1)))

                correct = torch.zeros(args.batch_size, device=args.device)
                total = torch.zeros(args.batch_size, device=args.device)
                for _, images, labels in task["eval_loader"]:
                    images = images.to(args.device, non_blocking=True)
                    labels = labels.to(args.device, non_blocking=True)
                    targets = build_paperflip_targets(labels, source_label)
                    counter = torch.zeros(
                        args.batch_size, labels.size(0), args.n_class, device=args.device
                    )
                    for params in params_list:
                        batch_id = 0
                        for layers in zip(*params):
                            out = hypertheft.eval_f(layers, images)
                            pred = out.argmax(-1)
                            counter[
                                batch_id,
                                torch.arange(labels.size(0), device=args.device),
                                pred,
                            ] += 1
                            batch_id += 1
                    voted_pred = counter.argmax(-1)
                    voted_correct = voted_pred.eq(targets.repeat(args.batch_size, 1))
                    threshold = math.floor(voter_count * 0.75)
                    mask = (counter > threshold).int().sum(-1)
                    correct += (voted_correct * mask).sum(-1)
                    total += mask.sum(-1)
                acc = torch.where(
                    total > 0, correct / total.clamp_min(1), torch.zeros_like(total)
                )
                all_acc.append(acc)
                label_breakdown.append(
                    {
                        "source_label": int(source_label),
                        "avg_acc": float(acc.mean().item()),
                        "max_acc": float(acc.max().item()),
                    }
                )
            acc = torch.cat(all_acc, 0)
            task_metrics["voters"].append(
                {
                    "num_voters": voter_count,
                    "max_acc": float(acc.max().item()),
                    "avg_acc": float(acc.mean().item()),
                    "label_breakdown": label_breakdown,
                }
            )
        task_summaries.append(task_metrics)
    return task_summaries


@torch.no_grad()
def run_sanity_check(args, tasks):
    task = tasks[0]
    batches_checked = 0
    rows = []
    start_label = 0
    for batch_index, (_, images, labels) in enumerate(task["train_loader"]):
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        source_label = int((start_label + batch_index) % 2)
        seed_batches = sample_sameclass_seed_batches(
            task["seed_cache_by_label_cpu"],
            batch_size=args.batch_size,
            n_seed=args.n_seed,
            source_label=source_label,
            device=args.device,
        )
        targets = build_paperflip_targets(labels, source_label)
        for classifier_id in range(min(4, args.batch_size)):
            rows.append(
                {
                    "classifier_id": int(classifier_id),
                    "seed_label": int(source_label),
                    "train_labels_head": labels[:8].detach().cpu().tolist(),
                    "paper_targets_head": targets[:8].detach().cpu().tolist(),
                }
            )
        batches_checked += 1
        if batches_checked >= int(args.sanity_batches):
            break
    payload = {
        "task_name": task["task_name"],
        "n_seed": int(args.n_seed),
        "batch_size": int(args.batch_size),
        "rule": "same-class seed batch; paper_target = 1 iff sample binary_label == source_label else 0",
        "examples": rows,
    }
    out_path = Path(args.output_dir) / "paperflip_sanity.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps(payload, indent=2))


def save_checkpoint(
    output_dir, epoch, hypertheft, optim_q, optim_g_list, schedulers, history
):
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
    args.trace_files = parse_trace_files_arg(args.trace_files)
    include_task_names = None
    if args.task_names:
        include_task_names = [
            name.strip() for name in args.task_names.split(",") if name.strip()
        ]
    if is_cuda_device(args.device) and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    if is_cuda_device(args.device):
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_layout = inspect_trace_layout(
        args.trace_root,
        args.mode,
        trace_w_override=args.trace_w,
        include_task_names=include_task_names,
        trace_files=args.trace_files,
    )
    args = build_runtime_args(args, trace_layout)
    num_voters = [int(x) for x in args.num_voters.split(",") if x.strip()]

    models = importlib.import_module(f"models.{args.target_model}")
    args.ngen = infer_ngen(models, args)

    config = {
        "args": vars(args),
        "trace_layout": trace_layout,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    hypertheft = models.HyperTheft(args)
    optim_q = torch.optim.Adam(
        hypertheft.mixer.parameters(), lr=args.lr, weight_decay=args.wd
    )
    optim_g_list = [
        torch.optim.Adam(gen.parameters(), lr=args.lr, weight_decay=args.wd)
        for gen in hypertheft.generator.as_list()
    ]
    scaler = GradScaler(enabled=bool(args.amp and is_cuda_device(args.device)))
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_q, T_max=max(args.epochs, 1)
        )
    ]
    schedulers.extend(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(args.epochs, 1)
        )
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
        input_shape=args.input_shape,
        granularity=args.granularity,
        task_limit=args.task_limit,
        include_task_names=include_task_names,
        trace_files=args.trace_files,
    )
    if not tasks:
        raise RuntimeError("No tasks were loaded for training")

    prepare_task_caches(
        tasks,
        seed_cache_limit=args.seed_cache_limit,
        batch_size=args.batch_size,
        n_seed=args.n_seed,
    )

    if args.sanity_only:
        run_sanity_check(args, tasks)
        return

    for epoch in range(start_epoch, args.epochs):
        train_summary = train_one_epoch(
            args=args,
            hypertheft=hypertheft,
            tasks=tasks,
            optimizers=(optim_q, optim_g_list),
            scaler=scaler,
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
            f"train_loss_mean={np.mean(train_losses):.4f}",
            flush=True,
        )
        if eval_summary is not None:
            eval_accs = [item["voters"][0]["avg_acc"] for item in eval_summary]
            print(f"epoch={epoch} eval_avg_acc_mean={np.mean(eval_accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
