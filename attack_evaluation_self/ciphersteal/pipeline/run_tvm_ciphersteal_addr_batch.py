#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("/path/to/tvm_ana_workspace")
GLOW_ROOT = Path("/path/to/glow_workspace")
DEFAULT_SESSION_PARENT = ROOT_DIR / "ciphersteal_addrtrace_runs"
PYTHON_BIN = ROOT_DIR / ".venv_ciphersteal" / "bin" / "python"

RELU_PATCH_ENV_ON = {
    "TVM_RELU_LOW12_PATCH": "1",
    "TVM_RELU_PATCH_BITS": "30",
    "TVM_RELU_PATCH_FIXED_BITS": "7",
    # The runtime parser uses strtoul(..., base=0), which does not accept a
    # 0b-prefixed literal. 0x75 is exactly 0b1110101.
    "TVM_RELU_PATCH_FIXED_VALUE": "0x75",
    "TVM_RELU_PATCH_INC": "3",
    "TVM_RELU_PATCH_POSITIVE": "0",
}

INPUT_DITHER_ENV_ON = {
    "TVM_INPUT_ZERO_DITHER": "random",
    "TVM_INPUT_ZERO_DITHER_LAYOUT": "NCHW",
    "TVM_INPUT_ZERO_DITHER_THRESH": "1",
    "TVM_INPUT_ZERO_DITHER_EPS_MIN": "1e-5",
    "TVM_INPUT_ZERO_DITHER_EPS_MAX": "2e-5",
    "TVM_INPUT_ZERO_DITHER_SILENT": "0",
}

REQUESTED_CONFIG = {
    "patchbits": 30,
    "fixedbits": 7,
    "fixedvalue_binary_requested": "0b1110101",
    "fixedvalue_runtime_env": "0x75",
    "fixedvalue_decimal": 117,
    "inc": 3,
    "positive": 0,
    "input_zero_dither": "random",
    "input_zero_dither_thresh": 1,
    "input_zero_dither_eps_min": "1e-5",
    "input_zero_dither_eps_max": "2e-5",
}

RELEVANT_ENV_KEYS = tuple(
    list(RELU_PATCH_ENV_ON.keys())
    + list(INPUT_DITHER_ENV_ON.keys())
    + [
        "GLOW_INPUT_ZERO_DITHER",
        "GLOW_INPUT_ZERO_DITHER_LAYOUT",
        "GLOW_INPUT_ZERO_DITHER_THRESH",
        "GLOW_INPUT_ZERO_DITHER_EPS_MIN",
        "GLOW_INPUT_ZERO_DITHER_EPS_MAX",
        "GLOW_INPUT_ZERO_DITHER_SILENT",
    ]
)


@dataclass(frozen=True)
class Spec:
    key: str
    model: str
    dataset: str
    backend: str
    input_family: str
    build_script: Path
    trace_script: Path
    runner_build_script: Path
    library_filename: str
    needs_constants_dir: bool
    runner_build_uses_out_dir: bool


@dataclass(frozen=True)
class InputEntry:
    sample_index: int
    label: int
    input_bin: str
    metadata_json: str


SPECS: dict[str, Spec] = {
    "lenet_mnist_native_vm": Spec(
        key="lenet_mnist_native_vm",
        model="lenet",
        dataset="mnist",
        backend="native_vm",
        input_family="mnist",
        build_script=ROOT_DIR / "lenet_mnist" / "lenet_mnist_tvm.py",
        trace_script=ROOT_DIR / "lenet_mnist" / "native_vm" / "trace" / "run_lenet_tvm_trace_and_analyze.sh",
        runner_build_script=ROOT_DIR / "lenet_mnist" / "native_vm" / "build_native_runner.sh",
        library_filename="lenet_mnist_tvm.so",
        needs_constants_dir=False,
        runner_build_uses_out_dir=False,
    ),
    "lenet_mnist_aot": Spec(
        key="lenet_mnist_aot",
        model="lenet",
        dataset="mnist",
        backend="aot",
        input_family="mnist",
        build_script=ROOT_DIR / "lenet_mnist" / "aot" / "export_lenet_mnist_aot.py",
        trace_script=ROOT_DIR / "lenet_mnist" / "aot" / "run_lenet_tvm_aot_trace_and_analyze.sh",
        runner_build_script=ROOT_DIR / "lenet_mnist" / "aot" / "build_aot_runner.sh",
        library_filename="lenet_mnist_tvm_aot_kernels.so",
        needs_constants_dir=True,
        runner_build_uses_out_dir=False,
    ),
    "vggnet_cifar_native_vm": Spec(
        key="vggnet_cifar_native_vm",
        model="vggnet",
        dataset="cifar",
        backend="native_vm",
        input_family="cifar",
        build_script=ROOT_DIR / "vggnet_cifar" / "vggnet_cifar_tvm.py",
        trace_script=ROOT_DIR / "vggnet_cifar" / "native_vm" / "trace" / "run_vggnet_tvm_trace_and_analyze.sh",
        runner_build_script=ROOT_DIR / "vggnet_cifar" / "native_vm" / "build_native_runner.sh",
        library_filename="vggnet_cifar_tvm.so",
        needs_constants_dir=False,
        runner_build_uses_out_dir=False,
    ),
    "vggnet_cifar_aot": Spec(
        key="vggnet_cifar_aot",
        model="vggnet",
        dataset="cifar",
        backend="aot",
        input_family="cifar",
        build_script=ROOT_DIR / "vggnet_cifar" / "aot" / "export_vggnet_cifar_aot.py",
        trace_script=ROOT_DIR / "vggnet_cifar" / "aot" / "run_vggnet_tvm_aot_trace_and_analyze.sh",
        runner_build_script=ROOT_DIR / "vggnet_cifar" / "aot" / "build_aot_runner.sh",
        library_filename="vggnet_cifar_tvm_aot_kernels.so",
        needs_constants_dir=True,
        runner_build_uses_out_dir=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and launch background TVM CipherSteal addr-trace runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Prepare inputs/builds and launch background workers.")
    launch.add_argument(
        "--session-dir",
        default=None,
        help="Explicit session directory. Defaults to ciphersteal_addrtrace_runs/<timestamp>.",
    )

    worker = subparsers.add_parser("worker", help="Run all samples/modes for one spec.")
    worker.add_argument("--session-dir", required=True, help="Prepared session directory.")
    worker.add_argument("--spec", required=True, choices=sorted(SPECS), help="Experiment spec key.")

    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="ascii")


def spec_to_json(spec: Spec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "model": spec.model,
        "dataset": spec.dataset,
        "backend": spec.backend,
        "input_family": spec.input_family,
        "build_script": str(spec.build_script),
        "trace_script": str(spec.trace_script),
        "runner_build_script": str(spec.runner_build_script),
        "library_filename": spec.library_filename,
        "needs_constants_dir": spec.needs_constants_dir,
        "runner_build_uses_out_dir": spec.runner_build_uses_out_dir,
    }


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def make_session_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (DEFAULT_SESSION_PARENT / timestamp_tag()).resolve()


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in RELEVANT_ENV_KEYS:
        env.pop(key, None)
    return env


def mode_env(mode: str) -> dict[str, str]:
    env = clean_env()
    if mode == "on":
        env.update(RELU_PATCH_ENV_ON)
        env.update(INPUT_DITHER_ENV_ON)
    return env


def render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_logged(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    cwd: Path = ROOT_DIR,
) -> None:
    ensure_parent(log_path)
    with log_path.open("w", encoding="ascii") as log_file:
        log_file.write(f"cwd={cwd}\n")
        log_file.write(f"cmd={render_cmd(cmd)}\n")
        for key in sorted(RELEVANT_ENV_KEYS):
            if key in env:
                log_file.write(f"{key}={env[key]}\n")
        log_file.write(f"begin={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log_file.flush()
        subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        log_file.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")


def generate_mnist_inputs(session_dir: Path) -> list[InputEntry]:
    raw_path = GLOW_ROOT / "dataset" / "mnist" / "MNIST" / "raw" / "t10k-images-idx3-ubyte"
    label_path = GLOW_ROOT / "dataset" / "mnist" / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"
    raw = raw_path.read_bytes()
    labels_raw = label_path.read_bytes()

    rows = int.from_bytes(raw[8:12], "big")
    cols = int.from_bytes(raw[12:16], "big")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, rows, cols)
    labels = np.frombuffer(labels_raw, dtype=np.uint8, offset=8)

    out_dir = session_dir / "generated_inputs" / "lenet_mnist"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[InputEntry] = []
    for idx in (0, 1):
        tensor = (images[idx].astype(np.float32) / 255.0).reshape(1, 1, rows, cols)
        bin_path = out_dir / f"mnist_test_idx{idx:04d}_nchw_f32.bin"
        meta_path = out_dir / f"mnist_test_idx{idx:04d}.json"
        tensor.tofile(bin_path)
        meta = {
            "dataset": "mnist",
            "sample_index": idx,
            "label": int(labels[idx]),
            "shape": [1, 1, rows, cols],
            "preprocess": "uint8 / 255.0 -> float32 NCHW",
            "source_images": str(raw_path),
            "source_labels": str(label_path),
        }
        write_json(meta_path, meta)
        entries.append(
            InputEntry(
                sample_index=idx,
                label=int(labels[idx]),
                input_bin=str(bin_path),
                metadata_json=str(meta_path),
            )
        )
    return entries


def generate_cifar_inputs(session_dir: Path) -> list[InputEntry]:
    raw_path = GLOW_ROOT / "dataset" / "cifar10" / "cifar-10-batches-py" / "test_batch"
    with raw_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = payload[b"data"]
    labels = payload[b"labels"]

    out_dir = session_dir / "generated_inputs" / "vggnet_cifar"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[InputEntry] = []
    for idx in (0, 1):
        tensor = (data[idx].reshape(3, 32, 32).astype(np.float32) / 255.0).reshape(1, 3, 32, 32)
        bin_path = out_dir / f"cifar_test_idx{idx:04d}_rgb32_nchw_f32.bin"
        meta_path = out_dir / f"cifar_test_idx{idx:04d}.json"
        tensor.tofile(bin_path)
        meta = {
            "dataset": "cifar10",
            "sample_index": idx,
            "label": int(labels[idx]),
            "shape": [1, 3, 32, 32],
            "preprocess": "uint8 / 255.0 -> float32 NCHW RGB",
            "source_test_batch": str(raw_path),
        }
        write_json(meta_path, meta)
        entries.append(
            InputEntry(
                sample_index=idx,
                label=int(labels[idx]),
                input_bin=str(bin_path),
                metadata_json=str(meta_path),
            )
        )
    return entries


def generate_inputs(session_dir: Path) -> dict[str, list[InputEntry]]:
    return {
        "mnist": generate_mnist_inputs(session_dir),
        "cifar": generate_cifar_inputs(session_dir),
    }


def build_out_dir(session_dir: Path, spec: Spec, mode: str) -> Path:
    return session_dir / "artifacts" / spec.key / mode


def runner_out_dir(session_dir: Path, spec: Spec) -> Path:
    return session_dir / "runners" / spec.key


def build_spec_artifacts(session_dir: Path, spec: Spec) -> dict[str, Any]:
    env_off = clean_env()
    env_on = clean_env()
    env_on["TVM_RELU_LOW12_PATCH"] = "1"

    build_logs: dict[str, str] = {}
    build_dirs: dict[str, str] = {}

    for mode, env in (("off", env_off), ("on", env_on)):
        out_dir = build_out_dir(session_dir, spec, mode)
        log_path = session_dir / "build_logs" / spec.key / f"{mode}.log"
        if spec.backend == "native_vm":
            cmd = [str(PYTHON_BIN), str(spec.build_script), "build", "--out-dir", str(out_dir)]
        else:
            cmd = [str(PYTHON_BIN), str(spec.build_script), "--out-dir", str(out_dir)]
        run_logged(cmd=cmd, env=env, log_path=log_path)
        build_logs[mode] = str(log_path)
        build_dirs[mode] = str(out_dir)

    runner_dir = runner_out_dir(session_dir, spec)
    runner_log = session_dir / "build_logs" / spec.key / "runner.log"
    runner_env = clean_env()
    if spec.runner_build_uses_out_dir:
        cmd = [
            "bash",
            str(spec.runner_build_script),
            str(build_out_dir(session_dir, spec, "on")),
            str(runner_dir),
        ]
    else:
        cmd = ["bash", str(spec.runner_build_script), str(runner_dir)]
    run_logged(cmd=cmd, env=runner_env, log_path=runner_log)

    runner_bin = runner_dir / (
        "lenet_mnist_tvm_native_runner"
        if spec.key == "lenet_mnist_native_vm"
        else "lenet_mnist_tvm_aot_runner"
        if spec.key == "lenet_mnist_aot"
        else "vggnet_cifar_tvm_native_runner"
        if spec.key == "vggnet_cifar_native_vm"
        else "vggnet_cifar_tvm_aot_runner"
    )

    return {
        "build_logs": build_logs,
        "build_dirs": build_dirs,
        "runner_log": str(runner_log),
        "runner_bin": str(runner_bin),
    }


def grep_file(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    result = subprocess.run(
        ["rg", "-q", needle, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def verify_build(session_dir: Path, spec: Spec) -> dict[str, Any]:
    report: dict[str, Any] = {"spec": spec.key, "modes": {}}
    for mode in ("off", "on"):
        out_dir = build_out_dir(session_dir, spec, mode)
        build_dir = out_dir / "build"
        library_path = build_dir / spec.library_filename
        marker_file = build_dir / ("lowered_relax.py" if spec.backend == "native_vm" else "tir_kernels.py")
        marker_text = marker_file.read_text(encoding="ascii", errors="ignore")
        has_relu_extern = "tvm_relu_low12_f32" in marker_text
        nm_result = subprocess.run(
            ["nm", "-D", str(library_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        has_symbol_ref = "tvm_relu_low12_f32" in nm_result.stdout
        expected = mode == "on"
        if has_symbol_ref != expected:
            raise RuntimeError(
                f"build verification failed for {spec.key}/{mode}: "
                f"has_relu_extern={has_relu_extern}, has_symbol_ref={has_symbol_ref}"
            )
        report["modes"][mode] = {
            "out_dir": str(out_dir),
            "library_path": str(library_path),
            "marker_file": str(marker_file),
            "has_tvm_relu_low12_f32_in_ir": has_relu_extern,
            "has_tvm_relu_low12_f32_symbol_ref": has_symbol_ref,
            "constants_dir": str(build_dir / "constants") if spec.needs_constants_dir else None,
        }
    return report


def prepare_session(session_dir: Path) -> dict[str, Any]:
    session_dir.mkdir(parents=True, exist_ok=True)
    inputs = generate_inputs(session_dir)
    build_reports: dict[str, Any] = {}
    verification_reports: dict[str, Any] = {}
    for spec in SPECS.values():
        build_reports[spec.key] = build_spec_artifacts(session_dir, spec)
        verification_reports[spec.key] = verify_build(session_dir, spec)

    manifest = {
        "session_dir": str(session_dir),
        "root_dir": str(ROOT_DIR),
        "glow_root": str(GLOW_ROOT),
        "python_bin": str(PYTHON_BIN),
        "requested_config": REQUESTED_CONFIG,
        "inputs": {
            family: [asdict(entry) for entry in entries] for family, entries in inputs.items()
        },
        "specs": {key: spec_to_json(spec) for key, spec in SPECS.items()},
        "builds": build_reports,
        "build_verification": verification_reports,
    }
    write_json(session_dir / "manifest.json", manifest)
    return manifest


def run_env_for_spec(session_dir: Path, spec: Spec, mode: str) -> dict[str, str]:
    env = mode_env(mode)
    env["PYTHON_BIN"] = str(PYTHON_BIN)
    env["WRITE_TAINT_TSV"] = "1"

    build_dir = build_out_dir(session_dir, spec, mode) / "build"
    env["LIBRARY_PATH"] = str(build_dir / spec.library_filename)
    if spec.backend == "native_vm":
        env["NATIVE_RUNNER_BIN"] = str(
            runner_out_dir(session_dir, spec)
            / (
                "lenet_mnist_tvm_native_runner"
                if spec.key == "lenet_mnist_native_vm"
                else "vggnet_cifar_tvm_native_runner"
            )
        )
    else:
        env["AOT_RUNNER_BIN"] = str(
            runner_out_dir(session_dir, spec)
            / (
                "lenet_mnist_tvm_aot_runner"
                if spec.key == "lenet_mnist_aot"
                else "vggnet_cifar_tvm_aot_runner"
            )
        )
        if spec.key == "vggnet_cifar_aot":
            env["RUNNER_SOURCE"] = str(build_dir / "generated_runner.cc")
    if spec.needs_constants_dir:
        env["CONSTANTS_DIR"] = str(build_dir / "constants")
    return env


def verify_run(spec: Spec, mode: str, run_dir: Path) -> dict[str, Any]:
    ipmap = run_dir / "paddrtrace.ip.txt"
    ref_stderr = run_dir / "reference.stderr.txt"
    pin_stderr = run_dir / "pin.stderr.txt"
    run_log = run_dir / "run.log"
    verify_json = run_dir / "verify.json"
    a2_json = run_dir / "a2_addr_change_bits.json"

    saw_relu_helper = grep_file(ipmap, "tvm_relu_low12_f32")
    saw_input_dither = grep_file(ref_stderr, "input_zero_dither=random") or grep_file(
        pin_stderr, "input_zero_dither=random"
    )

    expected = mode == "on"
    if saw_relu_helper != expected:
        raise RuntimeError(
            f"relu helper execution proof mismatch for {spec.key}/{mode} at {run_dir}: "
            f"saw_relu_helper={saw_relu_helper}"
        )
    if saw_input_dither != expected:
        raise RuntimeError(
            f"input dither execution proof mismatch for {spec.key}/{mode} at {run_dir}: "
            f"saw_input_dither={saw_input_dither}"
        )

    verify_payload = {
        "spec": spec.key,
        "mode": mode,
        "run_dir": str(run_dir),
        "run_log": str(run_log),
        "verify_json": str(verify_json),
        "a2_json": str(a2_json),
        "relu_helper_seen_in_ipmap": saw_relu_helper,
        "input_dither_logged": saw_input_dither,
        "requested_config": REQUESTED_CONFIG,
    }
    write_json(run_dir / "execution_proof.json", verify_payload)
    return verify_payload


def session_inputs(session_dir: Path, family: str) -> list[InputEntry]:
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="ascii"))
    return [InputEntry(**entry) for entry in manifest["inputs"][family]]


def worker_main(session_dir: Path, spec_key: str) -> int:
    spec = SPECS[spec_key]
    worker_log = session_dir / "worker_logs" / f"{spec.key}.log"
    ensure_parent(worker_log)

    entries = session_inputs(session_dir, spec.input_family)
    status: dict[str, Any] = {
        "spec": spec.key,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [],
    }

    with worker_log.open("w", encoding="ascii") as log_file:
        for entry in entries:
            for mode in ("off", "on"):
                out_root = session_dir / "runs" / spec.key / f"sample{entry.sample_index:04d}"
                run_dir = out_root / mode
                cmd = [
                    "bash",
                    str(spec.trace_script),
                    mode,
                    str(out_root),
                    entry.input_bin,
                ]
                env = run_env_for_spec(session_dir, spec, mode)
                log_file.write(f"cmd={render_cmd(cmd)}\n")
                log_file.write(f"input_bin={entry.input_bin}\n")
                log_file.write(f"label={entry.label}\n")
                for key in sorted(RELEVANT_ENV_KEYS):
                    if key in env:
                        log_file.write(f"{key}={env[key]}\n")
                log_file.write("\n")
                log_file.flush()
                subprocess.run(
                    cmd,
                    cwd=ROOT_DIR,
                    env=env,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                proof = verify_run(spec, mode, run_dir)
                proof["sample_index"] = entry.sample_index
                proof["label"] = entry.label
                status["runs"].append(proof)
                write_json(session_dir / "run_status" / spec.key / f"sample{entry.sample_index:04d}_{mode}.json", proof)

    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(session_dir / "worker_status" / f"{spec.key}.json", status)
    return 0


def launch_workers(session_dir: Path) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    launcher_log_dir = session_dir / "launcher_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    for spec in SPECS.values():
        log_path = launcher_log_dir / f"{spec.key}.log"
        log_file = log_path.open("w", encoding="ascii")
        cmd = [
            str(PYTHON_BIN),
            str(ROOT_DIR / "run_tvm_ciphersteal_addr_batch.py"),
            "worker",
            "--session-dir",
            str(session_dir),
            "--spec",
            spec.key,
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=clean_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        jobs[spec.key] = {
            "pid": proc.pid,
            "cmd": cmd,
            "log_path": str(log_path),
        }
        log_file.close()
    write_json(session_dir / "jobs.json", jobs)
    return jobs


def launch_main(session_dir: Path) -> int:
    manifest = prepare_session(session_dir)
    jobs = launch_workers(session_dir)
    summary = {
        "session_dir": manifest["session_dir"],
        "jobs": jobs,
        "requested_config": REQUESTED_CONFIG,
        "notes": [
            "Only the locally implemented TVM combinations are launched: lenet_mnist/vggnet_cifar x native_vm/aot.",
            "TVM_RELU_PATCH_FIXED_VALUE is passed as 0x75 because the runtime parser does not accept 0b1110101.",
        ],
    }
    write_json(session_dir / "launch_summary.json", summary)
    print(str(session_dir))
    return 0


def main() -> int:
    args = parse_args()
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"missing python environment: {PYTHON_BIN}")

    if args.command == "launch":
        return launch_main(make_session_dir(args.session_dir))
    if args.command == "worker":
        return worker_main(Path(args.session_dir).resolve(), args.spec)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
