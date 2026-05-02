from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


METRICS = ("stage_ns", "infer_ns", "total_ns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU-pinned persistent-server timing benchmarks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--outer-repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--cpu-list", default=None)
    parser.add_argument("--no-aslr-disable", action="store_true")
    return parser.parse_args()


def parse_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, errors="replace")


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_node_map() -> dict[int, int]:
    try:
        lines = run_text(["lscpu", "-p=cpu,node"]).splitlines()
    except Exception:
        return {}
    mapping = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        cpu, node = line.split(",", 1)
        mapping[int(cpu)] = int(node)
    return mapping


def pick_cpu(policy: dict[str, Any]) -> tuple[int, int]:
    mapping = cpu_node_map()
    if policy.get("force_cpu") is not None:
        cpu = int(policy["force_cpu"])
    else:
        preferred = parse_cpu_list(str(policy.get("preferred_cpus", "0")))
        cpu = preferred[0] if preferred else 0
    node = int(policy.get("force_numa_node", mapping.get(cpu, 0)) or 0)
    return cpu, node


def capture_environment(cpu: int, node: int) -> dict[str, Any]:
    env = {
        "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "host": platform.node(),
        "machine": platform.machine(),
        "selected_cpu": cpu,
        "selected_numa_node": node,
        "governor": read_text(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"),
        "frequency": read_text(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"),
        "boost": read_text("/sys/devices/system/cpu/cpufreq/boost"),
    }
    for key, command in {"lscpu": ["lscpu"], "numactl": ["numactl", "--hardware"]}.items():
        try:
            env[key] = run_text(command)
        except Exception as exc:
            env[key] = f"<failed: {exc}>"
    return env


def resolve_path(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else base / path)


def resolve_command(base: Path, command: list[Any]) -> list[str]:
    out = []
    for part in command:
        text = str(part)
        if text.startswith("-") or "/" not in text:
            out.append(text)
        else:
            out.append(resolve_path(base, text) or text)
    return out


def resolve_group(base: Path, group: dict[str, Any]) -> dict[str, Any]:
    out = dict(group)
    for key in ("executable", "weights", "input"):
        if key in out:
            out[key] = resolve_path(base, str(out[key]))
    if isinstance(out.get("command"), list):
        out["command"] = resolve_command(base, out["command"])
    return out


def selected_groups(config: dict[str, Any], base: Path, names: list[str] | None, allow_missing: bool) -> list[dict[str, Any]]:
    requested = set(names or [])
    groups, missing = [], []
    for raw in config.get("groups", []):
        if requested and raw.get("name") not in requested:
            continue
        if raw.get("status", "ready") != "ready":
            missing.append(str(raw.get("name")))
            continue
        group = resolve_group(base, raw)
        if not group.get("command") and not all(group.get(key) for key in ("executable", "weights", "input")):
            missing.append(str(group.get("name")))
            continue
        groups.append(group)
    unresolved = requested - {str(group["name"]) for group in groups}
    missing.extend(sorted(unresolved))
    if missing and not allow_missing:
        raise SystemExit(f"missing or not-ready groups: {', '.join(missing)}")
    if not groups:
        raise SystemExit("no benchmark groups selected")
    return groups


def sanitize_json_line(line: str) -> str:
    return re.sub(r'("output0":)([+-]?(?:nan|inf))(?=[}\s])', r"\1null", line, flags=re.IGNORECASE)


class Server:
    def __init__(self, group: dict[str, Any], cpu: int, node: int, disable_aslr: bool, use_numactl: bool, log_dir: Path):
        self.group = group
        self.log_path = log_dir / f"{group['name']}.stderr.log"
        self.log_file = self.log_path.open("w", encoding="utf-8")
        command: list[str] = []
        if disable_aslr and Path("/usr/bin/setarch").exists():
            command += ["/usr/bin/setarch", platform.machine(), "-R"]
        if shutil.which("taskset"):
            command += ["taskset", "-c", str(cpu)]
        if use_numactl and shutil.which("numactl"):
            command += ["numactl", f"--physcpubind={cpu}", f"--membind={node}"]
        command += [str(part) for part in group.get("command") or [group["executable"], group["weights"], group["input"], "--server"]]
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (group.get("env") or {}).items()})
        self.command = command
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file, text=True, bufsize=1, env=env)
        ready = self.read_json()
        if ready.get("type") != "ready":
            raise RuntimeError(f"{group['name']} did not become ready; see {self.log_path}")

    def read_json(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.group['name']} exited early; see {self.log_path}")
        return json.loads(sanitize_json_line(line))

    def run_once(self) -> dict[str, Any]:
        assert self.proc.stdin is not None
        self.proc.stdin.write("RUN\n")
        self.proc.stdin.flush()
        result = self.read_json()
        if result.get("type") != "result":
            raise RuntimeError(f"{self.group['name']} returned {result}")
        return result

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()
        finally:
            self.log_file.close()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "median": statistics.median(values) if values else float("nan"),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw: list[dict[str, Any]], groups: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    measured = [row for row in raw if row["phase"] == "measure"]
    summary = {"baseline_group": groups[0]["name"], "groups": {}}
    for group in groups:
        name = str(group["name"])
        rows = [row for row in measured if row["group"] == name]
        payload = {"metrics": {}, "repeat_median_summary": {}}
        for metric in metrics:
            values = [float(row[metric]) for row in rows if metric in row]
            payload["metrics"][metric] = metric_summary(values)
            repeat_medians = []
            for repeat in sorted({int(row["outer_repeat"]) for row in rows}):
                repeat_values = [float(row[metric]) for row in rows if int(row["outer_repeat"]) == repeat and metric in row]
                if repeat_values:
                    repeat_medians.append(statistics.median(repeat_values))
            payload["repeat_median_summary"][metric] = metric_summary(repeat_medians)
        summary["groups"][name] = payload
    baseline = summary["groups"][summary["baseline_group"]]
    for name, payload in summary["groups"].items():
        for metric in metrics:
            base = baseline["repeat_median_summary"][metric]["median"]
            cur = payload["repeat_median_summary"][metric]["median"]
            payload["repeat_median_summary"][metric]["overhead_vs_baseline_median"] = cur - base
            payload["repeat_median_summary"][metric]["overhead_vs_baseline_median_pct"] = 100.0 * (cur / base - 1.0) if base else float("nan")
    return summary


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    policy = dict(config.get("benchmark_policy", {}))
    cpu_policy = dict(config.get("cpu_policy", {}))
    for attr, key in [("warmup_runs", "warmup_runs"), ("rounds", "rounds"), ("outer_repeats", "outer_repeats"), ("seed", "seed")]:
        value = getattr(args, attr)
        if value is not None:
            policy[key] = value
    if args.cpu is not None:
        cpu_policy["force_cpu"] = args.cpu
    if args.cpu_list:
        cpu_policy["preferred_cpus"] = args.cpu_list
    groups = selected_groups(config, base, args.groups, args.allow_missing)
    warmup, rounds = int(policy.get("warmup_runs", 3)), int(policy.get("rounds", 10))
    repeats, seed = int(policy.get("outer_repeats", 1)), int(policy.get("seed", 1))
    metrics = list(config.get("metrics", METRICS))
    cpu, node = pick_cpu(cpu_policy)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "env.json").write_text(json.dumps(capture_environment(cpu, node), indent=2) + "\n", encoding="utf-8")
    raw: list[dict[str, Any]] = []
    rng = random.Random(seed)
    disable_aslr = bool(cpu_policy.get("disable_aslr", True)) and not args.no_aslr_disable
    use_numactl = bool(cpu_policy.get("use_numactl", True))
    logs = args.output_dir / "logs"
    logs.mkdir()
    for repeat in range(repeats):
        order = groups[:]
        rng.shuffle(order)
        for phase, count in [("warmup", warmup), ("measure", rounds)]:
            for group in order:
                server = Server(group, cpu, node, disable_aslr, use_numactl, logs)
                try:
                    for index in range(count):
                        result = server.run_once()
                        raw.append({"phase": phase, "outer_repeat": repeat, "run_index": index, "group": group["name"], **result})
                finally:
                    server.close()
    with (args.output_dir / "raw_runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in raw:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(raw, groups, metrics)
    summary.update({"framework": config.get("framework"), "variant": config.get("variant"), "selected_cpu": cpu, "selected_numa_node": node, "policy": policy})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for group, payload in summary["groups"].items():
        for metric, stats in payload["repeat_median_summary"].items():
            rows.append({"group": group, "metric": metric, **stats})
    write_csv(args.output_dir / "summary_repeat_medians.csv", rows)
    print(json.dumps({"output_dir": str(args.output_dir), "groups": [group["name"] for group in groups]}, indent=2))


if __name__ == "__main__":
    main()
