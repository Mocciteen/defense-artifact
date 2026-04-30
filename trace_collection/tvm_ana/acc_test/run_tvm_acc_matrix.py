#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("/path/to/tvm_ana_workspace")
ACC_TEST_DIR = ROOT_DIR / "acc_test"
FULL_MATRIX_DIR = ROOT_DIR / "full_matrix"
SESSION_PARENT = ACC_TEST_DIR / "sessions"
GLOW_ROOT = Path("/path/to/glow_workspace")
GLOW_ACC_TEST_DIR = GLOW_ROOT / "trace" / "acc_test"
GLOW_EVAL_PATH = GLOW_ACC_TEST_DIR / "common" / "eval_model_compiler_accuracy.py"
SUMMARY_MD = GLOW_ROOT / "evaluation_status_summary.md"

SUBSET_MANIFEST_CELEA_CHEST = (
    GLOW_ACC_TEST_DIR
    / "batch_runs"
    / "relu_coarse_parallel3_20260326_221057"
    / "subset_manifest_resnet_mobilenet_celea5000_chest5000.json"
)
SUBSET_MANIFEST_IMAGENET32 = (
    GLOW_ACC_TEST_DIR / "batch_runs" / "imagenet32_balanced_5000_seed1234.json"
)
SUBSET_MANIFEST_IMAGENET50_96 = (
    GLOW_ACC_TEST_DIR / "batch_runs" / "imagenet50_96_balanced_2500_seed1234.json"
)

CONFIG_ID = "b30_fb7_fv0b1110101_thr1_e1e-5_2e-5"


def load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if str(FULL_MATRIX_DIR) not in sys.path:
    sys.path.insert(0, str(FULL_MATRIX_DIR))

matrix_registry = load_module_from_path("matrix_registry", FULL_MATRIX_DIR / "matrix_registry.py")
run_matrix = load_module_from_path("run_matrix", FULL_MATRIX_DIR / "run_matrix.py")
glow_eval = load_module_from_path("glow_eval_model_compiler_accuracy", GLOW_EVAL_PATH)

ON_ENV = {**matrix_registry.RELU_PATCH_ENV_ON, **matrix_registry.INPUT_DITHER_ENV_ON}
RUN_ENV_KEYS = tuple(sorted(ON_ENV))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TVM accuracy/perf matrix for evaluation_status_summary.md sections 3 and 4."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available TV/TA specs.")
    list_parser.add_argument("--specs", default=None, help="Optional comma-separated subset.")

    launch_parser = subparsers.add_parser(
        "launch", help="Create a detached background manager for a TVM acc/perf session."
    )
    launch_parser.add_argument("--session-dir", default=None, help="Explicit session directory.")
    launch_parser.add_argument("--specs", default=None, help="Optional comma-separated subset.")
    launch_parser.add_argument("--parallelism", type=int, default=4)
    launch_parser.add_argument("--resume-existing", action="store_true")

    manager_parser = subparsers.add_parser(
        "manager", help="Prepare caches and schedule per-spec workers."
    )
    manager_parser.add_argument("--session-dir", required=True)
    manager_parser.add_argument("--specs", default=None, help="Optional comma-separated subset.")
    manager_parser.add_argument("--parallelism", type=int, default=4)
    manager_parser.add_argument("--resume-existing", action="store_true")

    eval_parser = subparsers.add_parser(
        "eval-spec", help="Evaluate one TVxx/TAxx spec into a per-spec summary."
    )
    eval_parser.add_argument("--session-dir", required=True)
    eval_parser.add_argument("--spec", required=True)
    eval_parser.add_argument("--resume-existing", action="store_true")

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Aggregate per-spec summaries into summary_all.json and summary.tsv."
    )
    aggregate_parser.add_argument("--session-dir", required=True)

    update_parser = subparsers.add_parser(
        "update-md", help="Fill sections 3 and 4 TVM rows in evaluation_status_summary.md."
    )
    update_parser.add_argument("--session-dir", required=True)
    update_parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)

    return parser.parse_args()


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def make_session_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (SESSION_PARENT / f"{timestamp_tag()}_{CONFIG_ID}").resolve()


def selected_spec_ids(spec_arg: str | None) -> list[str]:
    return run_matrix.selected_spec_ids(spec_arg)


def default_subset_manifest(dataset_name: str) -> Path | None:
    if dataset_name in {"celea", "chest"}:
        return SUBSET_MANIFEST_CELEA_CHEST
    if dataset_name == "imagenet32":
        return SUBSET_MANIFEST_IMAGENET32
    if dataset_name == "imagenet50_96":
        return SUBSET_MANIFEST_IMAGENET50_96
    return None


def case_model_spec(case: Any) -> Any:
    return glow_eval.MODEL_SPECS[case.model_family]


def case_dataset_spec(case: Any) -> Any:
    return case_model_spec(case).datasets[case.dataset]


def load_selection_cache(path: Path) -> list[int]:
    return [int(x) for x in np.load(path).tolist()]


def cache_paths_for_case(session_dir: Path, case_key: str) -> dict[str, Path]:
    root = session_dir / "base_cases" / case_key
    return {
        "root": root,
        "selected_indices": root / "selected_indices.npy",
        "selection": root / "selection.json",
        "pt_baseline": root / "pt_baseline.json",
    }


def prepare_case_cache(session_dir: Path, case: Any) -> dict[str, Any]:
    cache_paths = cache_paths_for_case(session_dir, case.key)
    cache_paths["root"].mkdir(parents=True, exist_ok=True)
    if (
        cache_paths["selected_indices"].is_file()
        and cache_paths["selection"].is_file()
        and cache_paths["pt_baseline"].is_file()
    ):
        selection = read_json(cache_paths["selection"])
        pt_baseline = read_json(cache_paths["pt_baseline"])
        selected_indices = load_selection_cache(cache_paths["selected_indices"])
        return {
            "case_key": case.key,
            "selected_indices_path": str(cache_paths["selected_indices"]),
            "selection": selection,
            "pt_baseline_path": str(cache_paths["pt_baseline"]),
            "pt_baseline": pt_baseline,
            "num_samples": len(selected_indices),
            "subset_manifest": selection.get("subset_manifest"),
        }

    model_spec = case_model_spec(case)
    dataset_spec = case_dataset_spec(case)
    adapter = glow_eval.build_dataset_adapter(dataset_spec)
    total_available = len(adapter)

    subset_manifest_path = default_subset_manifest(case.dataset)
    subset_indices_by_dataset: dict[str, list[int]] = {}
    subset_metadata_by_dataset: dict[str, dict[str, Any]] = {}
    if subset_manifest_path is not None:
        if not subset_manifest_path.is_file():
            raise FileNotFoundError(f"missing subset manifest: {subset_manifest_path}")
        subset_indices_by_dataset, subset_metadata_by_dataset = glow_eval.load_subset_manifest(
            subset_manifest_path
        )

    selected_indices, selection = glow_eval.build_dataset_selection(
        dataset_name=case.dataset,
        total_available=total_available,
        offset=0,
        limit=None,
        subset_indices_by_dataset=subset_indices_by_dataset,
        subset_metadata_by_dataset=subset_metadata_by_dataset,
        subset_manifest_path=subset_manifest_path,
    )
    np.save(cache_paths["selected_indices"], np.asarray(selected_indices, dtype=np.int64))
    write_json(cache_paths["selection"], selection)

    if subset_manifest_path is not None:
        pt_baseline = glow_eval.evaluate_pt_checkpoint_subset(
            model_spec,
            case.dataset,
            dataset_spec,
            selected_indices,
            batch_size=64,
            device="cpu",
        )
        pt_baseline["subset_manifest"] = str(subset_manifest_path)
    else:
        pt_baseline = glow_eval.load_pt_baseline(model_spec, case.dataset)
    write_json(cache_paths["pt_baseline"], pt_baseline)

    return {
        "case_key": case.key,
        "selected_indices_path": str(cache_paths["selected_indices"]),
        "selection": selection,
        "pt_baseline_path": str(cache_paths["pt_baseline"]),
        "pt_baseline": pt_baseline,
        "num_samples": len(selected_indices),
        "subset_manifest": str(subset_manifest_path) if subset_manifest_path is not None else None,
    }


def prepare_manifest(session_dir: Path, spec_ids: list[str], parallelism: int) -> dict[str, Any]:
    session_dir.mkdir(parents=True, exist_ok=True)
    spec_map = matrix_registry.runtime_specs()
    selected_specs = [spec_map[spec_id] for spec_id in spec_ids]

    selected_case_keys: list[str] = []
    for spec in selected_specs:
        if spec.base.key not in selected_case_keys:
            selected_case_keys.append(spec.base.key)

    base_cases = matrix_registry.base_case_map()
    evaluation: dict[str, Any] = {}
    for case_key in selected_case_keys:
        evaluation[case_key] = prepare_case_cache(session_dir, base_cases[case_key])

    effective_onnx = {
        case_key: run_matrix.prepare_effective_onnx(session_dir, base_cases[case_key])
        for case_key in selected_case_keys
    }
    generic_vm_runner = run_matrix.build_generic_vm_runner(session_dir)

    manifest = {
        "session_dir": str(session_dir),
        "root_dir": str(ROOT_DIR),
        "acc_test_dir": str(ACC_TEST_DIR),
        "full_matrix_dir": str(FULL_MATRIX_DIR),
        "glow_root": str(GLOW_ROOT),
        "config_id": CONFIG_ID,
        "parallelism": int(parallelism),
        "selected_spec_ids": spec_ids,
        "on_env": ON_ENV,
        "base_cases": {
            case.key: run_matrix.base_case_to_json(case)
            for case in matrix_registry.BASE_CASES
            if case.key in selected_case_keys
        },
        "specs": {spec_id: run_matrix.spec_to_json(spec_map[spec_id]) for spec_id in spec_ids},
        "evaluation": evaluation,
        "effective_onnx": effective_onnx,
        "common_artifacts": {
            "generic_vm_runner": str(generic_vm_runner),
        },
    }
    write_json(session_dir / "manifest.json", manifest)
    return manifest


class TVMStreamingSession:
    def __init__(
        self,
        *,
        command: list[str],
        output_elements: int,
        env_overrides: dict[str, str],
        stderr_path: Path,
    ) -> None:
        self.command = command
        self.output_elements = output_elements
        self.output_bytes = output_elements * np.dtype(np.float32).itemsize
        self.env_overrides = env_overrides
        self.stderr_path = stderr_path
        self.proc: subprocess.Popen[bytes] | None = None
        self.stderr_handle: Any | None = None
        self.startup_sec = 0.0
        self.forward_total_sec = 0.0
        self.shutdown_sec = 0.0
        self.infer_calls = 0
        self.peak_rss_kib: int | None = None
        self.current_rss_kib: int | None = None

    def __enter__(self) -> TVMStreamingSession:
        startup_begin = time.perf_counter()
        env = run_matrix.clean_env()
        env.update(self.env_overrides)
        self.stderr_handle = self.stderr_path.open("wb")
        self.proc = subprocess.Popen(
            self.command,
            cwd=ROOT_DIR,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            bufsize=0,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(f"failed to open streaming pipes for: {render_cmd(self.command)}")
        self.startup_sec = time.perf_counter() - startup_begin
        self._update_memory_status()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        pending_error: RuntimeError | None = None
        if self.proc is not None:
            shutdown_begin = time.perf_counter()
            self._update_memory_status()
            try:
                if self.proc.stdin is not None and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except BrokenPipeError:
                pass
            rc = self.proc.wait()
            self.shutdown_sec = time.perf_counter() - shutdown_begin
            if rc != 0 and exc_type is None:
                pending_error = RuntimeError(
                    f"TVM runner exited with code {rc}: {render_cmd(self.command)}\n"
                    f"stderr:\n{self._read_stderr()}"
                )
        if self.stderr_handle is not None:
            self.stderr_handle.close()
        if pending_error is not None:
            raise pending_error
        return False

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("streaming session is not active")

        payload = np.ascontiguousarray(tensor, dtype=np.float32)
        infer_begin = time.perf_counter()
        try:
            self.proc.stdin.write(payload.tobytes(order="C"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"TVM runner pipe closed while writing: {render_cmd(self.command)}\n"
                f"stderr:\n{self._read_stderr()}"
            ) from exc

        try:
            output_bytes = self._read_exact(self.output_bytes)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}\nstderr:\n{self._read_stderr()}") from exc

        self.forward_total_sec += time.perf_counter() - infer_begin
        self.infer_calls += 1
        self._update_memory_status()
        return np.frombuffer(output_bytes, dtype=np.float32).copy()

    def collect_perf_stats(self) -> dict[str, Any]:
        startup_to_end = self.startup_sec + self.forward_total_sec + self.shutdown_sec
        return {
            "startup_sec": self.startup_sec,
            "forward_total_sec": self.forward_total_sec,
            "forward_avg_sec": self.forward_total_sec / max(self.infer_calls, 1),
            "shutdown_sec": self.shutdown_sec,
            "startup_to_end_inference_sec": startup_to_end,
            "infer_calls": self.infer_calls,
            "peak_rss_kib": self.peak_rss_kib,
            "current_rss_kib": self.current_rss_kib,
        }

    def _read_exact(self, num_bytes: int) -> bytes:
        assert self.proc is not None and self.proc.stdout is not None
        chunks: list[bytes] = []
        remaining = num_bytes
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                rc = self.proc.poll()
                raise RuntimeError(
                    f"TVM runner ended early after {num_bytes - remaining}/{num_bytes} bytes "
                    f"(rc={rc})"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_stderr(self) -> str:
        if not self.stderr_path.exists():
            return ""
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")

    def _read_proc_status_memory(self) -> tuple[int | None, int | None]:
        if self.proc is None:
            return None, None
        status_path = Path("/proc") / str(self.proc.pid) / "status"
        if not status_path.exists():
            return None, None

        vm_rss_kib: int | None = None
        vm_hwm_kib: int | None = None
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    vm_rss_kib = int(parts[1])
            elif line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2:
                    vm_hwm_kib = int(parts[1])
        return vm_rss_kib, vm_hwm_kib

    def _update_memory_status(self) -> None:
        vm_rss_kib, vm_hwm_kib = self._read_proc_status_memory()
        if vm_rss_kib is not None:
            self.current_rss_kib = vm_rss_kib
        if vm_hwm_kib is not None:
            if self.peak_rss_kib is None:
                self.peak_rss_kib = vm_hwm_kib
            else:
                self.peak_rss_kib = max(self.peak_rss_kib, vm_hwm_kib)


def stream_command(
    *,
    spec: Any,
    runner_bin: Path,
    library_path: Path,
    constants_dir: Path | None,
) -> list[str]:
    if spec.runtime == "native_vm":
        shape_text = ",".join(str(dim) for dim in spec.base.input_cfg.input_shape)
        return [
            str(runner_bin),
            "--library",
            str(library_path),
            "--input-shape",
            shape_text,
            "--stream",
        ]
    if constants_dir is None:
        raise ValueError("constants_dir is required for AOT stream command")
    return [
        str(runner_bin),
        "--library",
        str(library_path),
        "--constants-dir",
        str(constants_dir),
        "--stream",
    ]


def single_run_command(
    *,
    spec: Any,
    runner_bin: Path,
    library_path: Path,
    constants_dir: Path | None,
    input_bin: Path,
    output_bin: Path,
    summary_json: Path,
    output_txt: Path,
) -> list[str]:
    if spec.runtime == "native_vm":
        shape_text = ",".join(str(dim) for dim in spec.base.input_cfg.input_shape)
        return [
            str(runner_bin),
            "--library",
            str(library_path),
            "--input-shape",
            shape_text,
            "--input-bin",
            str(input_bin),
            "--output-bin",
            str(output_bin),
            "--summary-json",
            str(summary_json),
            "--output-txt",
            str(output_txt),
        ]
    if constants_dir is None:
        raise ValueError("constants_dir is required for AOT single-run command")
    return [
        str(runner_bin),
        "--library",
        str(library_path),
        "--constants-dir",
        str(constants_dir),
        "--input-bin",
        str(input_bin),
        "--output-bin",
        str(output_bin),
        "--summary-json",
        str(summary_json),
        "--output-txt",
        str(output_txt),
    ]


def artifact_stats_for_mode(
    *,
    spec: Any,
    runner_bin: Path,
    library_path: Path,
    constants_dir: Path | None,
) -> dict[str, Any]:
    runner_bytes = runner_bin.stat().st_size
    library_bytes = library_path.stat().st_size
    constants_total_bytes = 0
    constants_files = 0
    if constants_dir is not None:
        for path in sorted(constants_dir.glob("*.bin")):
            constants_total_bytes += path.stat().st_size
            constants_files += 1

    payload = {
        "runner_size_bytes": int(runner_bytes),
        "library_size_bytes": int(library_bytes),
        "constants_total_bytes": int(constants_total_bytes),
        "constants_files": int(constants_files),
        "artifact_total_bytes": int(runner_bytes + library_bytes + constants_total_bytes),
    }
    if spec.runtime == "native_vm":
        payload["notes"] = "generic_vm_runner + compiled_vm_library"
    else:
        payload["notes"] = "generated_aot_runner + kernel_library + constant_blobs"
    return payload


def run_on_probe(
    *,
    spec: Any,
    runner_bin: Path,
    library_path: Path,
    constants_dir: Path | None,
    sample_input_bin: Path,
    probe_dir: Path,
) -> dict[str, Any]:
    probe_dir.mkdir(parents=True, exist_ok=True)
    cmd = single_run_command(
        spec=spec,
        runner_bin=runner_bin,
        library_path=library_path,
        constants_dir=constants_dir,
        input_bin=sample_input_bin,
        output_bin=probe_dir / "probe_output.bin",
        summary_json=probe_dir / "probe_summary.json",
        output_txt=probe_dir / "probe_output.txt",
    )
    env = run_matrix.clean_env()
    probe_env = dict(ON_ENV)
    probe_env["TVM_INPUT_ZERO_DITHER_SILENT"] = "0"
    env.update(probe_env)
    stderr_path = probe_dir / "probe.stderr.txt"
    stdout_path = probe_dir / "probe.stdout.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    return {
        "command": cmd,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_json": str(probe_dir / "probe_summary.json"),
        "input_dither_logged": "input_zero_dither=random" in stderr_text,
    }


def evaluate_mode(
    *,
    session_dir: Path,
    spec: Any,
    mode: str,
    runner_bin: Path,
    library_path: Path,
    constants_dir: Path | None,
    dataset_spec: Any,
    adapter: Any,
    selected_indices: list[int],
    baseline_predictions: np.ndarray | None,
    sample_input_bin: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    work_dir = session_dir / "eval_runs" / spec.spec_id / mode
    work_dir.mkdir(parents=True, exist_ok=True)

    env_overrides: dict[str, str] = {}
    execution_proof: dict[str, Any] = {}
    if mode == "on":
        env_overrides = dict(ON_ENV)
        env_overrides["TVM_INPUT_ZERO_DITHER_SILENT"] = "1"
        execution_proof = run_on_probe(
            spec=spec,
            runner_bin=runner_bin,
            library_path=library_path,
            constants_dir=constants_dir,
            sample_input_bin=sample_input_bin,
            probe_dir=work_dir / "probe",
        )

    stderr_path = work_dir / "runner_stream.stderr.txt"
    with TVMStreamingSession(
        command=stream_command(
            spec=spec,
            runner_bin=runner_bin,
            library_path=library_path,
            constants_dir=constants_dir,
        ),
        output_elements=dataset_spec.output_elements,
        env_overrides=env_overrides,
        stderr_path=stderr_path,
    ) as session:
        if dataset_spec.task_type == "single_label":
            result, predictions = glow_eval.evaluate_single_label_case(
                adapter=adapter,
                dataset_spec=dataset_spec,
                session=session,
                sample_indices=selected_indices,
                work_dir=work_dir,
                baseline_predictions=baseline_predictions,
            )
        else:
            result, predictions = glow_eval.evaluate_multi_label_case(
                adapter=adapter,
                dataset_spec=dataset_spec,
                session=session,
                sample_indices=selected_indices,
                work_dir=work_dir,
                baseline_binary_predictions=baseline_predictions,
            )
        perf_stats = session.collect_perf_stats()

    result["perf"] = perf_stats
    result["runner"] = str(runner_bin)
    result["library_path"] = str(library_path)
    result["constants_dir"] = str(constants_dir) if constants_dir is not None else None
    result["env"] = {key: env_overrides[key] for key in sorted(env_overrides)}
    result["execution_proof"] = execution_proof
    result["stderr_path"] = str(stderr_path)
    result["artifacts"] = artifact_stats_for_mode(
        spec=spec,
        runner_bin=runner_bin,
        library_path=library_path,
        constants_dir=constants_dir,
    )
    startup_to_end = float(perf_stats["startup_to_end_inference_sec"])
    forward_total = float(perf_stats["forward_total_sec"])
    num_samples = int(result["num_samples"])
    result["throughput_startup_to_end_sps"] = (
        num_samples / startup_to_end if startup_to_end > 0 else float("nan")
    )
    result["throughput_forward_only_sps"] = (
        num_samples / forward_total if forward_total > 0 else float("nan")
    )
    result["avg_forward_ms"] = float(perf_stats["forward_avg_sec"]) * 1000.0
    predictions_path = work_dir / "predictions.npy"
    np.save(predictions_path, predictions)
    result["predictions_path"] = str(predictions_path)
    return result, predictions


def worker_manifest(session_dir: Path) -> dict[str, Any]:
    return read_json(session_dir / "manifest.json")


def eval_spec_main(session_dir: Path, spec_id: str, resume_existing: bool) -> int:
    spec_summary_path = session_dir / "spec_summaries" / f"{spec_id}.json"
    if spec_summary_path.is_file():
        if resume_existing:
            return 0
        raise FileExistsError(f"spec summary already exists: {spec_summary_path}")

    manifest = worker_manifest(session_dir)
    spec = matrix_registry.runtime_specs()[spec_id]
    eval_cache = manifest["evaluation"][spec.base.key]
    selected_indices = load_selection_cache(Path(eval_cache["selected_indices_path"]))
    pt_baseline = read_json(Path(eval_cache["pt_baseline_path"]))
    vm_runner_bin = Path(manifest["common_artifacts"]["generic_vm_runner"])

    build_bundle = run_matrix.build_spec_artifacts(session_dir, spec, vm_runner_bin)
    runner_bin = Path(build_bundle["runner_info"]["runner_bin"])

    model_spec = case_model_spec(spec.base)
    dataset_spec = model_spec.datasets[spec.base.dataset]
    adapter = glow_eval.build_dataset_adapter(dataset_spec)
    sample_input_bin = (
        run_matrix.case_inputs_dir(session_dir, spec.base) / "sample0000_input_nchw_f32.bin"
    )
    if not sample_input_bin.is_file():
        run_matrix.generate_case_inputs(session_dir, spec.base)

    cases: dict[str, Any] = {}
    off_predictions: np.ndarray | None = None
    for mode in ("off", "on"):
        build_dir = session_dir / "artifacts" / spec.spec_id / mode / "build"
        library_path = build_dir / run_matrix.build_library_filename(spec)
        constants_dir = build_dir / "constants" if spec.runtime == "aot" else None
        case_result, predictions = evaluate_mode(
            session_dir=session_dir,
            spec=spec,
            mode=mode,
            runner_bin=runner_bin,
            library_path=library_path,
            constants_dir=constants_dir,
            dataset_spec=dataset_spec,
            adapter=adapter,
            selected_indices=selected_indices,
            baseline_predictions=off_predictions if mode != "off" else None,
            sample_input_bin=sample_input_bin,
        )
        case_result["delta_vs_pt"] = float(case_result["metric_value"]) - float(
            pt_baseline["metric_value"]
        )
        case_result["build_verification"] = build_bundle["build_verification"][mode]
        case_result["build_report"] = build_bundle["build_reports"][mode]
        cases[mode] = case_result
        if mode == "off":
            off_predictions = predictions

    off_metric = float(cases["off"]["metric_value"])
    for case_result in cases.values():
        case_result["delta_vs_off"] = float(case_result["metric_value"]) - off_metric

    payload = {
        "spec_id": spec.spec_id,
        "runtime": spec.runtime,
        "runtime_label": spec.runtime_label,
        "model_family": spec.base.model_family,
        "dataset": spec.base.dataset,
        "task_type": dataset_spec.task_type,
        "primary_metric_name": dataset_spec.primary_metric,
        "config_id": CONFIG_ID,
        "on_env": ON_ENV,
        "selection": eval_cache["selection"],
        "pt_baseline": pt_baseline,
        "build_verification": build_bundle["build_verification"],
        "build_reports": build_bundle["build_reports"],
        "runner_info": build_bundle["runner_info"],
        "cases": cases,
    }
    write_json(spec_summary_path, payload)
    write_json(session_dir / "worker_status" / f"{spec_id}.json", payload)
    return 0


def aggregate_main(session_dir: Path) -> int:
    spec_ids = worker_manifest(session_dir)["selected_spec_ids"]
    ordered: dict[str, Any] = {}
    for spec_id in spec_ids:
        summary_path = session_dir / "spec_summaries" / f"{spec_id}.json"
        if not summary_path.is_file():
            continue
        ordered[spec_id] = read_json(summary_path)

    summary_all = {
        "session_dir": str(session_dir),
        "config_id": CONFIG_ID,
        "selected_spec_ids": spec_ids,
        "specs": ordered,
    }
    write_json(session_dir / "summary_all.json", summary_all)

    rows = [
        "\t".join(
            [
                "spec_id",
                "runtime",
                "model_family",
                "dataset",
                "metric_name",
                "pt_baseline",
                "off_metric",
                "on_metric",
                "off_avg_forward_ms",
                "on_avg_forward_ms",
            ]
        )
    ]
    for spec_id in spec_ids:
        if spec_id not in ordered:
            continue
        summary = ordered[spec_id]
        rows.append(
            "\t".join(
                [
                    spec_id,
                    summary["runtime"],
                    summary["model_family"],
                    summary["dataset"],
                    summary["primary_metric_name"],
                    f"{float(summary['pt_baseline']['metric_value']):.6f}",
                    f"{float(summary['cases']['off']['metric_value']):.6f}",
                    f"{float(summary['cases']['on']['metric_value']):.6f}",
                    f"{float(summary['cases']['off']['avg_forward_ms']):.6f}",
                    f"{float(summary['cases']['on']['avg_forward_ms']):.6f}",
                ]
            )
        )
    write_text(session_dir / "summary.tsv", "\n".join(rows) + "\n")
    return 0


def fmt_delta(value: float) -> str:
    return f"{value:+0.6f}"


def fmt_sec(value: float) -> str:
    return f"{value:.4f} s"


def fmt_ms(value: float) -> str:
    return f"{value:.4f} ms"


def fmt_sps(value: float) -> str:
    return f"{value:.2f}"


def fmt_mib_from_kib(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value / 1024.0:.2f} MiB"


def fmt_mib_from_bytes(value: int) -> str:
    return f"{value / (1024.0 * 1024.0):.2f} MiB"


def fmt_bytes_delta(value: int) -> str:
    return f"{value:+d} B"


def build_tvm_row_maps(session_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    summary_all = read_json(session_dir / "summary_all.json")
    specs = summary_all["specs"]
    sec3_rows: dict[str, str] = {}
    sec4_rows: dict[str, str] = {}

    for spec_id in matrix_registry.ordered_spec_ids():
        summary = specs.get(spec_id)
        if summary is None:
            sec3_rows[spec_id] = f"| {spec_id} |  |  |  |  |  |  |"
            sec4_rows[spec_id] = f"| {spec_id} |  |  |  |  |  |  |  |  |  |  |  |  |  |"
            continue
        off_case = summary["cases"]["off"]
        on_case = summary["cases"]["on"]
        pt = float(summary["pt_baseline"]["metric_value"])
        off_metric = float(off_case["metric_value"])
        on_metric = float(on_case["metric_value"])
        sec3_rows[spec_id] = (
            f"| {spec_id} | {pt:.6f} | {off_metric:.6f} | `{CONFIG_ID}` | {on_metric:.6f} | "
            f"{fmt_delta(float(on_case['delta_vs_pt']))} | "
            f"{fmt_delta(float(on_case['delta_vs_off']))} |"
        )

        off_total = float(off_case["perf"]["startup_to_end_inference_sec"])
        on_total = float(on_case["perf"]["startup_to_end_inference_sec"])
        overhead = ((on_total - off_total) / off_total * 100.0) if off_total > 0 else float("nan")
        off_avg = float(off_case["avg_forward_ms"])
        on_avg = float(on_case["avg_forward_ms"])
        off_tp = float(off_case["throughput_startup_to_end_sps"])
        on_tp = float(on_case["throughput_startup_to_end_sps"])
        off_rss = fmt_mib_from_kib(off_case["perf"].get("peak_rss_kib"))
        on_rss = fmt_mib_from_kib(on_case["perf"].get("peak_rss_kib"))
        off_artifacts = int(off_case["artifacts"]["artifact_total_bytes"])
        on_artifacts = int(on_case["artifacts"]["artifact_total_bytes"])
        samples = int(off_case["num_samples"])
        remark = (
            f"samples={samples}; {summary['runtime']} "
            f"{off_case['artifacts'].get('notes', '').strip()}"
        ).strip()
        sec4_rows[spec_id] = (
            f"| {spec_id} | {fmt_sec(off_total)} | {fmt_sec(on_total)} | {overhead:+0.2f}% | "
            f"{fmt_ms(off_avg)} | {fmt_ms(on_avg)} | {fmt_sps(off_tp)} | {fmt_sps(on_tp)} | "
            f"{off_rss} | {on_rss} | {fmt_mib_from_bytes(off_artifacts)} | "
            f"{fmt_mib_from_bytes(on_artifacts)} | {fmt_bytes_delta(on_artifacts - off_artifacts)} | "
            f"{remark} |"
        )
    return sec3_rows, sec4_rows


def replace_table_rows(text: str, row_map: dict[str, str]) -> str:
    lines = text.splitlines()
    out_lines: list[str] = []
    for line in lines:
        replaced = False
        for exp_id, row in row_map.items():
            prefix = f"| {exp_id} |"
            if line.startswith(prefix):
                out_lines.append(row)
                replaced = True
                break
        if not replaced:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def replace_table_rows_in_section(text: str, section_title: str, row_map: dict[str, str]) -> str:
    lines = text.splitlines()
    start_idx: int | None = None
    end_idx = len(lines)
    for idx, line in enumerate(lines):
        if line.strip() == section_title:
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"missing section: {section_title}")
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].startswith("## "):
            end_idx = idx
            break

    replaced_section = replace_table_rows("\n".join(lines[start_idx:end_idx]) + "\n", row_map)
    new_lines = lines[:start_idx] + replaced_section.splitlines() + lines[end_idx:]
    return "\n".join(new_lines) + "\n"


def update_md_main(session_dir: Path, summary_md: Path) -> int:
    sec3_rows, sec4_rows = build_tvm_row_maps(session_dir)
    original = summary_md.read_text(encoding="utf-8")
    updated = replace_table_rows_in_section(original, "## 3. 任务效用记录表", sec3_rows)
    updated = replace_table_rows_in_section(updated, "## 4. 系统代价记录表", sec4_rows)
    summary_md.write_text(updated, encoding="utf-8")
    return 0


def launch_workers(
    session_dir: Path,
    spec_ids: list[str],
    parallelism: int,
    resume_existing: bool,
) -> dict[str, Any]:
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
            log_file = log_path.open("w", encoding="utf-8")
            cmd = [
                str(matrix_registry.PYTHON_BIN),
                str(ACC_TEST_DIR / "run_tvm_acc_matrix.py"),
                "eval-spec",
                "--session-dir",
                str(session_dir),
                "--spec",
                spec_id,
            ]
            if resume_existing:
                cmd.append("--resume-existing")
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT_DIR,
                env=run_matrix.clean_env(),
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


def manager_main(session_dir: Path, spec_ids: list[str], parallelism: int, resume_existing: bool) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    manager_status_path = session_dir / "manager_status.json"
    write_json(
        manager_status_path,
        {
            "session_dir": str(session_dir),
            "state": "preparing",
            "selected_spec_ids": spec_ids,
            "parallelism": int(parallelism),
            "resume_existing": bool(resume_existing),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    prepare_manifest(session_dir, spec_ids, parallelism)
    write_json(
        manager_status_path,
        {
            "session_dir": str(session_dir),
            "state": "running",
            "selected_spec_ids": spec_ids,
            "parallelism": int(parallelism),
            "resume_existing": bool(resume_existing),
            "manifest": str(session_dir / "manifest.json"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    completed = launch_workers(session_dir, spec_ids, parallelism, resume_existing)
    failures = sorted(spec_id for spec_id, result in completed.items() if result["returncode"] != 0)
    aggregate_main(session_dir)

    payload = {
        "session_dir": str(session_dir),
        "state": "finished" if not failures else "finished_with_failures",
        "selected_spec_ids": spec_ids,
        "parallelism": int(parallelism),
        "resume_existing": bool(resume_existing),
        "manifest": str(session_dir / "manifest.json"),
        "completed": completed,
        "failures": failures,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(manager_status_path, payload)
    return 0 if not failures else 1


def launch_main(session_dir: Path, spec_ids: list[str], parallelism: int, resume_existing: bool) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    manager_log = session_dir / "manager.log"
    cmd = [
        str(matrix_registry.PYTHON_BIN),
        str(ACC_TEST_DIR / "run_tvm_acc_matrix.py"),
        "manager",
        "--session-dir",
        str(session_dir),
        "--parallelism",
        str(parallelism),
        "--specs",
        ",".join(spec_ids),
    ]
    if resume_existing:
        cmd.append("--resume-existing")
    log_file = manager_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=run_matrix.clean_env(),
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
            "resume_existing": bool(resume_existing),
            "selected_spec_ids": spec_ids,
            "config_id": CONFIG_ID,
            "on_env": ON_ENV,
        },
    )
    print(str(session_dir))
    return 0


def list_main(spec_ids: list[str]) -> int:
    specs = matrix_registry.runtime_specs()
    for spec_id in spec_ids:
        spec = specs[spec_id]
        print(
            f"{spec.spec_id}\t{spec.runtime}\t{spec.base.model_family}\t"
            f"{spec.base.dataset}\t{spec.base.metric}"
        )
    return 0


def main() -> int:
    args = parse_args()
    if not matrix_registry.PYTHON_BIN.exists():
        raise FileNotFoundError(f"missing python environment: {matrix_registry.PYTHON_BIN}")

    spec_ids = selected_spec_ids(getattr(args, "specs", None))

    if args.command == "list":
        return list_main(spec_ids)
    if args.command == "launch":
        return launch_main(make_session_dir(args.session_dir), spec_ids, args.parallelism, args.resume_existing)
    if args.command == "manager":
        return manager_main(Path(args.session_dir).resolve(), spec_ids, args.parallelism, args.resume_existing)
    if args.command == "eval-spec":
        return eval_spec_main(Path(args.session_dir).resolve(), args.spec, args.resume_existing)
    if args.command == "aggregate":
        return aggregate_main(Path(args.session_dir).resolve())
    if args.command == "update-md":
        return update_md_main(Path(args.session_dir).resolve(), args.summary_md.resolve())
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
