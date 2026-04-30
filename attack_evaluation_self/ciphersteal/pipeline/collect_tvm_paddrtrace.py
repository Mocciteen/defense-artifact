#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("/path/to/high_leakage_workspace")
SESSION_PARENT = ROOT_DIR / "tvm_sessions"
SOURCE_SESSION = Path(
    "/path/to/tvm_ana_workspace/full_matrix/sessions/20260403_193243_fixcliprelu6_taintblock16"
)
TVM_ROOT = Path("/path/to/tvm_ana_workspace")
PYTHON_BIN = TVM_ROOT / ".venv_ciphersteal" / "bin" / "python"
PIN_BIN = Path("/path/to/glow_workspace/pin/pin")
PINTOOL = Path(
    "/path/to/glow_workspace/build_release_all/glow/pintrace/obj-intel64/paddrtrace.so"
)

DEFAULT_SPEC_IDS = [
    *(f"TV{i:02d}" for i in range(1, 18)),
    *(f"TA{i:02d}" for i in range(1, 18)),
]

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

TVM_ENV_OFF = {
    "TVM_RELU_LOW12_PATCH": "0",
    "TVM_INPUT_ZERO_DITHER": "0",
}

TVM_ENV_ON = {
    "TVM_RELU_LOW12_PATCH": "1",
    "TVM_RELU_PATCH_BITS": "30",
    "TVM_RELU_PATCH_FIXED_BITS": "7",
    "TVM_RELU_PATCH_FIXED_VALUE": "0x75",
    "TVM_RELU_PATCH_INC": "3",
    "TVM_RELU_PATCH_POSITIVE": "0",
    "TVM_INPUT_ZERO_DITHER": "random",
    "TVM_INPUT_ZERO_DITHER_LAYOUT": "NCHW",
    "TVM_INPUT_ZERO_DITHER_THRESH": "1",
    "TVM_INPUT_ZERO_DITHER_EPS_MIN": "1e-5",
    "TVM_INPUT_ZERO_DITHER_EPS_MAX": "2e-5",
    "TVM_INPUT_ZERO_DITHER_SILENT": "0",
}

RELEVANT_ENV_KEYS = tuple(
    set(TVM_ENV_OFF)
    | set(TVM_ENV_ON)
    | {
        "GLOW_INPUT_ZERO_DITHER",
        "GLOW_INPUT_ZERO_DITHER_LAYOUT",
        "GLOW_INPUT_ZERO_DITHER_THRESH",
        "GLOW_INPUT_ZERO_DITHER_EPS_MIN",
        "GLOW_INPUT_ZERO_DITHER_EPS_MAX",
        "GLOW_INPUT_ZERO_DITHER_SILENT",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect paddrtrace runs for TV01-TV17 and TA01-TA17 using the existing exact TVM session."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List selected specs.")
    list_parser.add_argument("--specs", default=None)

    launch_parser = subparsers.add_parser("launch", help="Create a session and detach a manager.")
    launch_parser.add_argument("--session-dir", default=None)
    launch_parser.add_argument("--specs", default=None)
    launch_parser.add_argument("--parallelism", type=int, default=4)

    manager_parser = subparsers.add_parser("manager", help="Run the session manager.")
    manager_parser.add_argument("--session-dir", required=True)
    manager_parser.add_argument("--specs", default=None)
    manager_parser.add_argument("--parallelism", type=int, default=4)

    worker_parser = subparsers.add_parser("worker", help="Run one spec.")
    worker_parser.add_argument("--session-dir", required=True)
    worker_parser.add_argument("--spec", required=True)

    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="ascii", errors="ignore")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def make_session_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (SESSION_PARENT / timestamp_tag()).resolve()


def render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def wrap_disable_aslr(cmd: list[str]) -> list[str]:
    if sys.platform != "linux":
        return cmd
    setarch = shutil.which("setarch")
    if setarch is None:
        return cmd
    return [setarch, platform.machine(), "-R", *cmd]


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in RELEVANT_ENV_KEYS:
        env.pop(key, None)
    return env


def mode_env(mode: str) -> dict[str, str]:
    env = clean_env()
    env.update(TVM_ENV_ON if mode == "on" else TVM_ENV_OFF)
    return env


def selected_spec_ids(spec_arg: str | None) -> list[str]:
    if spec_arg is None:
        return list(DEFAULT_SPEC_IDS)
    selected = [part.strip() for part in spec_arg.split(",") if part.strip()]
    unknown = [spec_id for spec_id in selected if spec_id not in DEFAULT_SPEC_IDS]
    if unknown:
        raise ValueError(f"unknown specs: {', '.join(unknown)}")
    return selected


def file_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for chunk in handle:
            if needle in chunk:
                return True
    return False


def compare_outputs(reference_bin: Path, trace_bin: Path, out_json: Path) -> None:
    reference = np.fromfile(reference_bin, dtype=np.float32)
    traced = np.fromfile(trace_bin, dtype=np.float32)
    if reference.shape != traced.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)} vs trace {tuple(traced.shape)}"
        )
    diff = traced - reference
    same_argmax = None
    same_sign_pattern = None
    cosine_similarity = None
    if reference.size > 1:
        same_argmax = bool(int(np.argmax(reference)) == int(np.argmax(traced)))
    same_sign_pattern = bool(np.array_equal(reference >= 0.0, traced >= 0.0))
    ref_norm = float(np.linalg.norm(reference))
    trace_norm = float(np.linalg.norm(traced))
    if ref_norm > 0.0 and trace_norm > 0.0:
        cosine_similarity = float(np.dot(reference, traced) / (ref_norm * trace_norm))
    payload = {
        "reference_npy": str(reference_bin),
        "trace_npy": str(trace_bin),
        "shape": list(reference.shape),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "allclose_rtol_1e-6_atol_1e-6": bool(np.allclose(reference, traced, rtol=1e-6, atol=1e-6)),
        "allclose_rtol_1e-3_atol_1e-2": bool(np.allclose(reference, traced, rtol=1e-3, atol=1e-2)),
        "same_argmax": same_argmax,
        "same_sign_pattern": same_sign_pattern,
        "cosine_similarity": cosine_similarity,
    }
    payload["semantic_match"] = bool(payload["same_sign_pattern"]) or bool(
        payload["same_argmax"] and payload["cosine_similarity"] is not None and payload["cosine_similarity"] >= 0.999
    )
    payload["verification_passed"] = bool(
        payload["allclose_rtol_1e-3_atol_1e-2"] or payload["semantic_match"]
    )
    write_json(out_json, payload)
    return payload


def source_manifest() -> dict[str, Any]:
    return read_json(SOURCE_SESSION / "manifest.json")


def source_worker_status(spec_id: str) -> dict[str, Any]:
    return read_json(SOURCE_SESSION / "worker_status" / f"{spec_id}.json")


def source_build_status(spec_id: str, mode: str) -> dict[str, Any]:
    return read_json(SOURCE_SESSION / "build_status" / spec_id / f"{mode}.json")


def validate_trace(run_dir: Path) -> dict[str, Any]:
    trace_bin = run_dir / "paddrtrace.bin"
    ipmap = run_dir / "paddrtrace.ip.txt"
    checks = {
        "trace_bin_exists": trace_bin.exists(),
        "trace_bin_nonempty": trace_bin.exists() and trace_bin.stat().st_size > 0,
        "ipmap_exists": ipmap.exists(),
        "ipmap_nonempty": ipmap.exists() and ipmap.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"paddrtrace validation failed for {run_dir}: {checks}")
    return checks


def run_reference_and_pin(
    *,
    spec_id: str,
    runtime: str,
    runner_bin: Path,
    library_path: Path,
    constants_dir: Path | None,
    input_bin: str,
    input_shape: list[int],
    run_dir: Path,
    mode: str,
) -> None:
    env = mode_env(mode)
    run_dir.mkdir(parents=True, exist_ok=True)

    ref_bin = run_dir / "reference_output.bin"
    ref_json = run_dir / "reference_summary.json"
    ref_txt = run_dir / "reference_output.txt"
    ref_stdout = run_dir / "reference.stdout.txt"
    ref_stderr = run_dir / "reference.stderr.txt"
    trace_bin = run_dir / "trace_output.bin"
    trace_json = run_dir / "trace_summary.json"
    trace_txt = run_dir / "trace_output.txt"
    pin_stdout = run_dir / "pin.stdout.txt"
    pin_stderr = run_dir / "pin.stderr.txt"
    paddr_bin = run_dir / "paddrtrace.bin"
    paddr_ip = run_dir / "paddrtrace.ip.txt"
    verify_json = run_dir / "verify.json"
    execution_proof = run_dir / "execution_proof.json"
    run_log = run_dir / "run.log"
    cmd_txt = run_dir / "cmd.txt"

    if runtime == "native_vm":
        shape_text = ",".join(str(dim) for dim in input_shape)
        reference_cmd = [
            str(runner_bin),
            "--library",
            str(library_path),
            "--input-bin",
            input_bin,
            "--input-shape",
            shape_text,
            "--output-bin",
            str(ref_bin),
            "--summary-json",
            str(ref_json),
            "--output-txt",
            str(ref_txt),
        ]
        trace_inner_cmd = [
            str(runner_bin),
            "--library",
            str(library_path),
            "--input-bin",
            input_bin,
            "--input-shape",
            shape_text,
            "--output-bin",
            str(trace_bin),
            "--summary-json",
            str(trace_json),
            "--output-txt",
            str(trace_txt),
        ]
    else:
        if constants_dir is None:
            raise ValueError(f"constants_dir missing for {spec_id}/{mode}")
        reference_cmd = [
            str(runner_bin),
            "--library",
            str(library_path),
            "--constants-dir",
            str(constants_dir),
            "--input-bin",
            input_bin,
            "--output-bin",
            str(ref_bin),
            "--summary-json",
            str(ref_json),
            "--output-txt",
            str(ref_txt),
        ]
        trace_inner_cmd = [
            str(runner_bin),
            "--library",
            str(library_path),
            "--constants-dir",
            str(constants_dir),
            "--input-bin",
            input_bin,
            "--output-bin",
            str(trace_bin),
            "--summary-json",
            str(trace_json),
            "--output-txt",
            str(trace_txt),
        ]

    pin_cmd = [
        str(PIN_BIN),
        "-t",
        str(PINTOOL),
        "-stack-depth",
        "0",
        "-no-paddr",
        "1",
        "-taint-seed-mode",
        "file",
        "-taint-file",
        input_bin,
        "-taint-only",
        "1",
        "-o",
        str(paddr_bin),
        "-m",
        str(paddr_ip),
        "--",
        *trace_inner_cmd,
    ]

    wrapped_reference_cmd = wrap_disable_aslr(reference_cmd)
    wrapped_pin_cmd = wrap_disable_aslr(pin_cmd)

    write_text(
        run_log,
        "\n".join(
            [
                f"begin={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                f"spec_id={spec_id}",
                f"mode={mode}",
                f"runtime={runtime}",
                f"runner_bin={runner_bin}",
                f"library_path={library_path}",
                f"constants_dir={constants_dir if constants_dir else ''}",
                f"input_bin={input_bin}",
                f"disable_aslr={'1' if wrapped_reference_cmd != reference_cmd else '0'}",
                *[f"{key}={value}" for key, value in sorted(env.items()) if key.startswith("TVM_")],
            ]
        )
        + "\n",
    )
    write_text(
        cmd_txt,
        render_cmd(wrapped_reference_cmd) + "\n" + render_cmd(wrapped_pin_cmd) + "\n",
    )

    with ref_stdout.open("w", encoding="ascii") as stdout_file, ref_stderr.open(
        "w", encoding="ascii"
    ) as stderr_file:
        subprocess.run(
            wrapped_reference_cmd,
            cwd=TVM_ROOT,
            env={**clean_env(), **env},
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    with pin_stdout.open("w", encoding="ascii") as stdout_file, pin_stderr.open(
        "w", encoding="ascii"
    ) as stderr_file:
        subprocess.run(
            wrapped_pin_cmd,
            cwd=TVM_ROOT,
            env={**clean_env(), **env},
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    verify_payload = compare_outputs(ref_bin, trace_bin, verify_json)
    trace_validation = validate_trace(run_dir)

    saw_relu_helper = file_contains(paddr_ip, "tvm_relu_low12_f32") or file_contains(
        paddr_ip, "tvm_relu6_low12_f32"
    )
    saw_input_dither = file_contains(ref_stderr, "input_zero_dither=random") or file_contains(
        pin_stderr, "input_zero_dither=random"
    )
    expected = mode == "on"
    if saw_relu_helper != expected:
        raise RuntimeError(
            f"relu/relu6 helper execution proof mismatch for {spec_id}/{mode}: {saw_relu_helper}"
        )
    if saw_input_dither != expected:
        raise RuntimeError(
            f"input dither execution proof mismatch for {spec_id}/{mode}: {saw_input_dither}"
        )

    write_json(
        execution_proof,
        {
            "spec_id": spec_id,
            "mode": mode,
            "run_dir": str(run_dir),
            "run_log": str(run_log),
            "verify_json": str(verify_json),
            "trace_kind": "paddrtrace",
            "trace_bin": str(paddr_bin),
            "trace_ipmap": str(paddr_ip),
            "trace_validation": trace_validation,
            "relu_helper_seen_in_ipmap": saw_relu_helper,
            "input_dither_logged": saw_input_dither,
            "verification_passed": bool(verify_payload["verification_passed"]),
            "requested_config": REQUESTED_CONFIG,
            "source_session": str(SOURCE_SESSION),
        },
    )
    with run_log.open("a", encoding="ascii") as handle:
        handle.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")


def worker_main(session_dir: Path, spec_id: str) -> int:
    worker = source_worker_status(spec_id)
    manifest = source_manifest()
    base_case = worker["base_case"]
    inputs = manifest["inputs"][base_case]
    runtime = worker["build_reports"]["off"]["build_meta"]["runtime"] if "runtime" in worker["build_reports"]["off"]["build_meta"] else source_build_status(spec_id, "off")["runtime"]
    runner_bin = Path(worker["runner_info"]["runner_bin"])
    runner_input_shape = worker["runner_info"].get("input_shape")

    runs: list[dict[str, Any]] = []
    for mode in ("off", "on"):
        build_status = source_build_status(spec_id, mode)
        library_path = Path(build_status["library_path"])
        constants_dir = Path(build_status["constants_dir"]) if build_status["constants_dir"] else None
        input_shape = build_status["build_meta"]["input_shape"]
        if runner_input_shape is not None:
            input_shape = runner_input_shape
        for entry in inputs:
            sample_tag = f"sample{int(entry['sample_index']):04d}"
            run_dir = session_dir / "runs" / spec_id / sample_tag / mode
            run_reference_and_pin(
                spec_id=spec_id,
                runtime=runtime,
                runner_bin=runner_bin,
                library_path=library_path,
                constants_dir=constants_dir,
                input_bin=entry["input_bin"],
                input_shape=input_shape,
                run_dir=run_dir,
                mode=mode,
            )
            proof = read_json(run_dir / "execution_proof.json")
            proof["sample_index"] = int(entry["sample_index"])
            proof["label"] = entry["label"]
            proof["metadata_json"] = entry["metadata_json"]
            write_json(
                session_dir / "execution_proofs" / spec_id / f"{sample_tag}_{mode}.json",
                proof,
            )
            runs.append(proof)

    payload = {
        "spec_id": spec_id,
        "base_case": base_case,
        "runtime": runtime,
        "source_session": str(SOURCE_SESSION),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": runs,
    }
    write_json(session_dir / "worker_status" / f"{spec_id}.json", payload)
    return 0


def list_main(spec_ids: list[str]) -> int:
    for spec_id in spec_ids:
        worker = source_worker_status(spec_id)
        print(f"{spec_id}\t{worker['base_case']}")
    return 0


def launch_workers(session_dir: Path, spec_ids: list[str], parallelism: int) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    launcher_log_dir = session_dir / "launcher_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(spec_ids)
    active: dict[str, tuple[subprocess.Popen[Any], io.TextIOWrapper, Path]] = {}
    completed: dict[str, Any] = {}

    while pending or active:
        while pending and len(active) < parallelism:
            spec_id = pending.pop(0)
            log_path = launcher_log_dir / f"{spec_id}.log"
            log_file = log_path.open("w", encoding="ascii")
            cmd = [
                str(PYTHON_BIN),
                str(Path(__file__).resolve()),
                "worker",
                "--session-dir",
                str(session_dir),
                "--spec",
                spec_id,
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT_DIR,
                env=clean_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active[spec_id] = (proc, log_file, log_path)
            jobs[spec_id] = {
                "pid": proc.pid,
                "cmd": cmd,
                "log_path": str(log_path),
                "state": "running",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            write_json(session_dir / "jobs.json", jobs)

        time.sleep(5)

        finished_ids: list[str] = []
        for spec_id, (proc, log_file, log_path) in active.items():
            returncode = proc.poll()
            if returncode is None:
                continue
            log_file.close()
            jobs[spec_id]["state"] = "finished"
            jobs[spec_id]["returncode"] = int(returncode)
            jobs[spec_id]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            completed[spec_id] = {
                "returncode": int(returncode),
                "log_path": str(log_path),
            }
            finished_ids.append(spec_id)
            write_json(session_dir / "jobs.json", jobs)
        for spec_id in finished_ids:
            active.pop(spec_id)

    return completed


def launch_main(session_dir: Path, spec_ids: list[str], parallelism: int) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    manager_log = session_dir / "manager.log"
    cmd = [
        str(PYTHON_BIN),
        str(Path(__file__).resolve()),
        "manager",
        "--session-dir",
        str(session_dir),
        "--parallelism",
        str(parallelism),
        "--specs",
        ",".join(spec_ids),
    ]
    log_file = manager_log.open("w", encoding="ascii")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=clean_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()

    write_json(
        session_dir / "launch_summary.json",
        {
            "session_dir": str(session_dir),
            "manager_pid": proc.pid,
            "manager_log": str(manager_log),
            "parallelism": int(parallelism),
            "selected_spec_ids": spec_ids,
            "source_session": str(SOURCE_SESSION),
            "requested_config": REQUESTED_CONFIG,
            "trace_kind": "paddrtrace",
        },
    )
    print(str(session_dir))
    return 0


def manager_main(session_dir: Path, spec_ids: list[str], parallelism: int) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    manager_status_path = session_dir / "manager_status.json"
    write_json(
        manager_status_path,
        {
            "session_dir": str(session_dir),
            "state": "running",
            "selected_spec_ids": spec_ids,
            "parallelism": int(parallelism),
            "source_session": str(SOURCE_SESSION),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    write_json(
        session_dir / "manifest.json",
        {
            "session_dir": str(session_dir),
            "source_session": str(SOURCE_SESSION),
            "selected_spec_ids": spec_ids,
            "parallelism": int(parallelism),
            "trace_kind": "paddrtrace",
            "requested_config": REQUESTED_CONFIG,
        },
    )

    completed = launch_workers(session_dir, spec_ids, parallelism)
    failures = sorted([spec_id for spec_id, result in completed.items() if result["returncode"] != 0])
    payload = {
        "session_dir": str(session_dir),
        "state": "finished" if not failures else "finished_with_failures",
        "selected_spec_ids": spec_ids,
        "parallelism": int(parallelism),
        "source_session": str(SOURCE_SESSION),
        "completed": completed,
        "failures": failures,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(manager_status_path, payload)
    return 0 if not failures else 1


def main() -> int:
    args = parse_args()
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"missing python environment: {PYTHON_BIN}")
    if not SOURCE_SESSION.exists():
        raise FileNotFoundError(f"missing source session: {SOURCE_SESSION}")
    if not PIN_BIN.exists():
        raise FileNotFoundError(f"missing pin binary: {PIN_BIN}")
    if not PINTOOL.exists():
        raise FileNotFoundError(f"missing pintool: {PINTOOL}")

    spec_ids = selected_spec_ids(getattr(args, "specs", None))

    if args.command == "list":
        return list_main(spec_ids)
    if args.command == "launch":
        return launch_main(make_session_dir(args.session_dir), spec_ids, args.parallelism)
    if args.command == "manager":
        return manager_main(Path(args.session_dir).resolve(), spec_ids, args.parallelism)
    if args.command == "worker":
        return worker_main(Path(args.session_dir).resolve(), args.spec)
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
