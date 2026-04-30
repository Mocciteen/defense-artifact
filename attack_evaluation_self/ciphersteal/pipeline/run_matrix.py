#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import pickle
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image

from matrix_registry import (
    BASE_CASES,
    CHEST_LABELS,
    DATA_ROOT,
    FULL_MATRIX_DIR,
    GLOW_ROOT,
    GENERIC_VM_RUNNER_SRC,
    IMAGENET50_96_ROOT,
    INPUT_DITHER_ENV_ON,
    PIN_BIN,
    PINTOOL,
    PYTHON_BIN,
    RELU_PATCH_ENV_ON,
    RELEVANT_ENV_KEYS,
    REQUESTED_CONFIG,
    ROOT_DIR,
    SESSION_PARENT,
    TVM_BUILD_DIR,
    TVM_HOME,
    BaseCase,
    InputConfig,
    RuntimeSpec,
    base_case_map,
    ordered_spec_ids,
    runtime_specs,
)


TARGET = "llvm"
OPT_LEVEL = 3


@dataclass(frozen=True)
class InputEntry:
    sample_index: int
    input_bin: str
    metadata_json: str
    label: Any


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    @property
    def num_elements(self) -> int:
        return int(np.prod(self.shape))


@dataclass(frozen=True)
class ConstantEntry:
    var_name: str
    filename: str
    spec: TensorSpec


@dataclass(frozen=True)
class InputRef:
    kind: str
    name: str


@dataclass(frozen=True)
class CallEntry:
    output_name: str
    kernel_name: str
    inputs: tuple[InputRef, ...]
    output_spec: TensorSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and manage the full TVM TV01-TV17 / TA01-TA17 addr-trace matrix."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List the full TVM matrix registry.")
    list_parser.add_argument("--specs", default=None, help="Optional comma-separated subset.")

    launch_parser = subparsers.add_parser(
        "launch", help="Create a session and start a detached background manager."
    )
    launch_parser.add_argument("--session-dir", default=None, help="Explicit session directory.")
    launch_parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Maximum number of concurrent worker subprocesses.",
    )
    launch_parser.add_argument("--specs", default=None, help="Optional comma-separated subset.")

    manager_parser = subparsers.add_parser(
        "manager", help="Prepare the session and schedule worker subprocesses."
    )
    manager_parser.add_argument("--session-dir", required=True)
    manager_parser.add_argument("--specs", default=None, help="Optional comma-separated subset.")
    manager_parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Maximum number of concurrent worker subprocesses.",
    )

    build_parser = subparsers.add_parser(
        "build-mode", help="Build one TVxx/TAxx spec for one mode in an isolated process."
    )
    build_parser.add_argument("--session-dir", required=True)
    build_parser.add_argument("--spec", required=True)
    build_parser.add_argument("--mode", required=True, choices=("off", "on"))

    worker_parser = subparsers.add_parser("worker", help="Build and run one TVxx/TAxx spec.")
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
    return json.loads(path.read_text(encoding="ascii"))


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
    if mode == "on":
        env.update(RELU_PATCH_ENV_ON)
        env.update(INPUT_DITHER_ENV_ON)
    return env


@contextlib.contextmanager
def scoped_relevant_env(env_updates: dict[str, str]):
    saved: dict[str, str | None] = {key: os.environ.get(key) for key in RELEVANT_ENV_KEYS}
    try:
        for key in RELEVANT_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in env_updates.items():
            os.environ[key] = value
        yield
    finally:
        for key in RELEVANT_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def selected_spec_ids(spec_arg: str | None) -> list[str]:
    all_specs = runtime_specs()
    if spec_arg is None:
        return ordered_spec_ids()
    selected = [part.strip() for part in spec_arg.split(",") if part.strip()]
    unknown = [spec_id for spec_id in selected if spec_id not in all_specs]
    if unknown:
        raise ValueError(f"unknown specs: {', '.join(unknown)}")
    return selected


def input_cfg_to_json(cfg: InputConfig) -> dict[str, Any]:
    return {
        "source_family": cfg.source_family,
        "input_shape": list(cfg.input_shape),
        "resize_hw": list(cfg.resize_hw) if cfg.resize_hw is not None else None,
        "source_mode": cfg.source_mode,
        "grayscale_to_rgb": cfg.grayscale_to_rgb,
        "mean": list(cfg.mean),
        "std": list(cfg.std),
        "assumptions": list(cfg.assumptions),
    }


def base_case_to_json(case: BaseCase) -> dict[str, Any]:
    return {
        "ordinal": case.ordinal,
        "key": case.key,
        "model_family": case.model_family,
        "dataset": case.dataset,
        "task_kind": case.task_kind,
        "metric": case.metric,
        "onnx_path": str(case.onnx_path),
        "input_cfg": input_cfg_to_json(case.input_cfg),
    }


def spec_to_json(spec: RuntimeSpec) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "runtime": spec.runtime,
        "runtime_label": spec.runtime_label,
        "base_case_key": spec.base.key,
        "model_family": spec.base.model_family,
        "dataset": spec.base.dataset,
        "task_kind": spec.base.task_kind,
        "metric": spec.base.metric,
        "onnx_path": str(spec.base.onnx_path),
    }


def prepare_tvm_imports() -> None:
    build_dir = TVM_BUILD_DIR
    if not build_dir.exists():
        raise FileNotFoundError(f"missing TVM build directory: {build_dir}")
    python_dir = TVM_HOME / "python"
    ffi_python_dir = TVM_HOME / "3rdparty" / "tvm-ffi" / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"missing TVM python directory: {python_dir}")
    if not ffi_python_dir.exists():
        raise FileNotFoundError(f"missing TVM FFI python directory: {ffi_python_dir}")

    os.environ.setdefault("TVM_LIBRARY_PATH", str(build_dir))
    for path in (python_dir, ffi_python_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_tvm_modules():
    prepare_tvm_imports()
    import tvm
    from tvm import relax
    from tvm.relax.expr import (
        Call,
        Constant,
        DataflowVar,
        GlobalVar,
        Tuple,
        TupleGetItem,
        Var,
        VarBinding,
    )
    from tvm.relax.frontend.onnx import from_onnx

    return (
        tvm,
        relax,
        from_onnx,
        Call,
        Constant,
        DataflowVar,
        GlobalVar,
        Tuple,
        TupleGetItem,
        Var,
        VarBinding,
    )


def module_to_text(mod: Any, *, show_meta: bool = False) -> str:
    if hasattr(mod, "script"):
        try:
            return mod.script(show_meta=show_meta)
        except TypeError:
            return mod.script()
    if hasattr(mod, "astext"):
        return mod.astext()
    return str(mod)


def tensor_spec_from_sinfo(sinfo: Any) -> TensorSpec:
    shape_expr = getattr(sinfo, "shape", None)
    if shape_expr is None or not hasattr(shape_expr, "values"):
        raise TypeError(f"expected static tensor shape, got {type(sinfo)}")
    shape = tuple(int(dim) for dim in shape_expr.values)
    dtype = str(getattr(sinfo, "dtype", ""))
    if not dtype:
        raise TypeError(f"missing dtype on tensor struct info: {sinfo}")
    return TensorSpec(shape=shape, dtype=dtype)


def expr_key(expr: Any) -> Any:
    vid = getattr(expr, "vid", None)
    if vid is not None:
        return vid
    return expr


def extract_kernel_module(lowered_mod: Any, tvm: Any) -> tuple[Any, list[str]]:
    kernel_funcs: dict[Any, Any] = {}
    kernel_names: list[str] = []
    for gvar, func in lowered_mod.functions.items():
        if isinstance(func, tvm.tirx.function.PrimFunc):
            kernel_name = gvar.name_hint
            kernel_funcs[gvar] = func.with_attr("global_symbol", kernel_name)
            kernel_names.append(kernel_name)
    if not kernel_funcs:
        raise RuntimeError("no TIR kernels found in lowered module")
    return tvm.IRModule(kernel_funcs, attrs=lowered_mod.attrs), kernel_names


def collect_lowered_call_sequence(
    lowered_mod: Any,
) -> tuple[Any, TensorSpec, Any, list[tuple[Any, str, TensorSpec, tuple[Any, ...]]]]:
    (
        _,
        _,
        _,
        Call,
        _,
        DataflowVar,
        GlobalVar,
        Tuple,
        TupleGetItem,
        Var,
        VarBinding,
    ) = load_tvm_modules()

    main = lowered_mod["main"]
    if len(main.params) != 1:
        raise ValueError(f"expected exactly one input parameter, got {len(main.params)}")
    input_var = main.params[0]
    input_spec = tensor_spec_from_sinfo(input_var.struct_info)

    aliases: dict[Any, Any] = {}
    resolve_cache: dict[Any, Any] = {}
    calls: list[tuple[Any, str, TensorSpec, tuple[Any, ...]]] = []

    def resolve_expr(expr: Any, resolving: set[int] | None = None) -> Any:
        if isinstance(expr, Tuple):
            return tuple(resolve_expr(field, resolving) for field in expr.fields)
        if isinstance(expr, TupleGetItem):
            tuple_value = resolve_expr(expr.tuple_value, resolving)
            if not isinstance(tuple_value, tuple):
                raise TypeError(f"expected tuple source for TupleGetItem, got {type(tuple_value)}")
            return resolve_expr(tuple_value[int(expr.index)], resolving)
        if isinstance(expr, (Var, DataflowVar)):
            key = expr_key(expr)
            if key in resolve_cache:
                return resolve_cache[key]
            target = aliases.get(key)
            if target is None:
                return expr
            if resolving is None:
                resolving = set()
            if key in resolving:
                return expr
            resolving.add(key)
            try:
                resolved = resolve_expr(target, resolving)
            finally:
                resolving.remove(key)
            resolve_cache[key] = resolved
            return resolved
        return expr

    for block in main.body.blocks:
        for binding in block.bindings:
            if not isinstance(binding, VarBinding):
                raise TypeError(f"unsupported binding type: {type(binding)}")
            value = binding.value
            if isinstance(value, Tuple):
                key = expr_key(binding.var)
                aliases[key] = value
                resolve_cache.pop(key, None)
                continue
            if isinstance(value, TupleGetItem):
                key = expr_key(binding.var)
                aliases[key] = value
                resolve_cache.pop(key, None)
                continue
            if isinstance(value, (Var, DataflowVar)):
                key = expr_key(binding.var)
                aliases[key] = value
                resolve_cache.pop(key, None)
                continue
            if not isinstance(value, Call):
                raise TypeError(f"unsupported binding value: {type(value)}")
            if str(value.op) != "Op(relax.call_tir)":
                raise TypeError(f"unsupported call op: {value.op}")
            if len(value.args) != 2:
                raise ValueError("call_tir is expected to have two arguments")

            kernel_gvar = value.args[0]
            args_tuple = value.args[1]
            if not isinstance(kernel_gvar, GlobalVar):
                raise TypeError(f"unexpected call_tir target: {type(kernel_gvar)}")
            if not isinstance(args_tuple, Tuple):
                raise TypeError(f"unexpected call_tir args tuple: {type(args_tuple)}")

            calls.append(
                (
                    binding.var,
                    kernel_gvar.name_hint,
                    tensor_spec_from_sinfo(value.sinfo_args[0]),
                    tuple(resolve_expr(arg) for arg in args_tuple.fields),
                )
            )

    output_expr = resolve_expr(main.body.body)
    return input_var, input_spec, output_expr, calls


def collect_runner_plan(
    lowered_mod: Any,
) -> tuple[str, TensorSpec, str, TensorSpec, list[ConstantEntry], list[CallEntry]]:
    _, _, _, _, Constant, DataflowVar, _, _, _, Var, _ = load_tvm_modules()
    input_var, input_spec, output_expr, resolved_calls = collect_lowered_call_sequence(lowered_mod)
    input_name = input_var.name_hint
    if not isinstance(output_expr, (Var, DataflowVar)):
        raise TypeError(f"unsupported main output expression: {type(output_expr)}")

    constants: list[ConstantEntry] = []
    calls: list[CallEntry] = []
    const_index = 0
    used_names = {input_name}
    canonical_names: dict[Any, str] = {expr_key(input_var): input_name}

    def canonical_var_name(var: Any) -> str:
        key = expr_key(var)
        existing = canonical_names.get(key)
        if existing is not None:
            return existing
        base = getattr(var, "name_hint", "") or "tensor"
        name = base
        suffix = 1
        while name in used_names:
            name = f"{base}_{suffix:03d}"
            suffix += 1
        canonical_names[key] = name
        used_names.add(name)
        return name

    for output_var, kernel_name, output_spec, resolved_args in resolved_calls:
        input_refs: list[InputRef] = []
        for arg in resolved_args:
            if isinstance(arg, (Var, DataflowVar)):
                input_refs.append(InputRef(kind="tensor", name=canonical_var_name(arg)))
                continue
            if isinstance(arg, Constant):
                array = arg.data.numpy()
                if str(array.dtype) != "float32":
                    raise TypeError(f"unsupported constant dtype: {array.dtype}")
                const_name = f"const_{const_index:02d}"
                filename = f"{const_index:02d}_{const_name}.bin"
                constants.append(
                    ConstantEntry(
                        var_name=const_name,
                        filename=filename,
                        spec=TensorSpec(
                            shape=tuple(int(dim) for dim in array.shape),
                            dtype=str(array.dtype),
                        ),
                    )
                )
                input_refs.append(InputRef(kind="constant", name=const_name))
                const_index += 1
                continue
            raise TypeError(f"unsupported call_tir argument: {type(arg)}")

        calls.append(
            CallEntry(
                output_name=canonical_var_name(output_var),
                kernel_name=kernel_name,
                inputs=tuple(input_refs),
                output_spec=output_spec,
            )
        )

    output_name = canonical_var_name(output_expr)
    output_spec = tensor_spec_from_sinfo(output_expr.struct_info)
    return input_name, input_spec, output_name, output_spec, constants, calls


def export_constant_blobs(*, lowered_mod: Any, constants_dir: Path) -> list[dict[str, Any]]:
    _, _, _, _, Constant, _, _, _, _, _, _ = load_tvm_modules()
    constants_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, resolved_calls = collect_lowered_call_sequence(lowered_mod)
    entries: list[dict[str, Any]] = []
    const_index = 0
    for _, _, _, resolved_args in resolved_calls:
        for arg in resolved_args:
            if not isinstance(arg, Constant):
                continue
            array = np.asarray(arg.data.numpy(), dtype=np.float32)
            const_name = f"const_{const_index:02d}"
            filename = f"{const_index:02d}_{const_name}.bin"
            path = constants_dir / filename
            array.tofile(path)
            entries.append(
                {
                    "index": int(const_index),
                    "var_name": const_name,
                    "filename": filename,
                    "shape": [int(dim) for dim in array.shape],
                    "dtype": str(array.dtype),
                    "num_elements": int(array.size),
                    "nbytes": int(array.nbytes),
                }
            )
            const_index += 1
    return entries


def shape_literal(shape: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(int(dim)) for dim in shape) + "}"


def shape_json(shape: tuple[int, ...]) -> str:
    return ", ".join(str(int(dim)) for dim in shape)


def generate_runner_source(
    *,
    input_name: str,
    input_spec: TensorSpec,
    output_name: str,
    output_spec: TensorSpec,
    constants: list[ConstantEntry],
    calls: list[CallEntry],
    library_filename: str,
) -> str:
    kernels: list[str] = []
    for call in calls:
        if call.kernel_name not in kernels:
            kernels.append(call.kernel_name)

    const_lines = []
    for const in constants:
        const_lines.append(
            "    Tensor {name} = TensorFromFile(JoinPath(args.constants_dir, {filename}), {elems}, {shape});".format(
                name=const.var_name,
                filename=json.dumps(const.filename),
                elems=const.spec.num_elements,
                shape=shape_literal(const.spec.shape),
            )
        )

    tensor_alloc_lines = []
    call_lines = []
    tensor_storage_names: dict[str, str] = {input_name: input_name}
    for index, call in enumerate(calls):
        storage_name = f"tensor_{index:04d}"
        tensor_storage_names[call.output_name] = storage_name
        tensor_alloc_lines.append(
            f"    Tensor {storage_name} = EmptyTensor({shape_literal(call.output_spec.shape)});"
        )

        arg_names: list[str] = []
        for ref in call.inputs:
            if ref.kind == "constant":
                arg_names.append(ref.name)
                continue
            if ref.kind != "tensor":
                raise ValueError(f"unsupported input ref kind: {ref.kind}")
            storage_ref = tensor_storage_names.get(ref.name)
            if storage_ref is None:
                raise KeyError(f"missing tensor storage for {ref.name}")
            arg_names.append(storage_ref)
        call_lines.append(f"    {call.kernel_name}({', '.join(arg_names)}, {storage_name});")

    output_storage_name = tensor_storage_names.get(output_name, output_name)

    kernel_decl_lines = [
        f'    Function {kernel} = RequireFunction(kernels, "{kernel}");' for kernel in kernels
    ]

    return f"""#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/input_zero_dither.h>
#include <tvm/runtime/relu_low12_patch.h>
#include <tvm/runtime/tensor.h>

namespace {{

using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::runtime::DataType;
using tvm::runtime::Tensor;

constexpr int64_t kInputElems = {input_spec.num_elements};
constexpr int64_t kOutputElems = {output_spec.num_elements};

struct Args {{
  std::string library_path;
  std::string constants_dir;
  std::string input_bin;
  std::string output_bin;
  std::string output_txt;
  std::string summary_json;
  bool stream_mode{{false}};
}};

void Usage(const char* argv0) {{
  std::cerr << "Usage: " << argv0
            << " --library <{library_filename}>"
            << " --constants-dir <constants_dir>"
            << " [--input-bin <input_f32.bin>]"
            << " [--output-bin <output_f32.bin>]"
            << " [--output-txt <output.txt>]"
            << " [--summary-json <summary.json>]"
            << " [--stream]\\n";
}}

std::string RequireValue(int& i, int argc, char** argv, const char* flag) {{
  if (i + 1 >= argc) {{
    throw std::runtime_error(std::string("missing value for ") + flag);
  }}
  ++i;
  return argv[i];
}}

Args ParseArgs(int argc, char** argv) {{
  Args out;
  for (int i = 1; i < argc; ++i) {{
    std::string arg = argv[i];
    if (arg == "--library") {{
      out.library_path = RequireValue(i, argc, argv, "--library");
    }} else if (arg == "--constants-dir") {{
      out.constants_dir = RequireValue(i, argc, argv, "--constants-dir");
    }} else if (arg == "--input-bin") {{
      out.input_bin = RequireValue(i, argc, argv, "--input-bin");
    }} else if (arg == "--output-bin") {{
      out.output_bin = RequireValue(i, argc, argv, "--output-bin");
    }} else if (arg == "--output-txt") {{
      out.output_txt = RequireValue(i, argc, argv, "--output-txt");
    }} else if (arg == "--summary-json") {{
      out.summary_json = RequireValue(i, argc, argv, "--summary-json");
    }} else if (arg == "--stream") {{
      out.stream_mode = true;
    }} else if (arg == "-h" || arg == "--help") {{
      Usage(argv[0]);
      std::exit(0);
    }} else {{
      throw std::runtime_error("unknown argument: " + arg);
    }}
  }}

  if (out.library_path.empty() || out.constants_dir.empty()) {{
    throw std::runtime_error("missing required arguments");
  }}
  if (!out.stream_mode && (out.input_bin.empty() || out.output_bin.empty())) {{
    throw std::runtime_error("missing required arguments");
  }}
  return out;
}}

std::vector<float> ReadFloat32File(const std::string& path, int64_t expected_elems) {{
  std::ifstream in(path, std::ios::binary);
  if (!in) {{
    throw std::runtime_error("failed to open input file: " + path);
  }}
  in.seekg(0, std::ios::end);
  std::streamoff size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size < 0) {{
    throw std::runtime_error("failed to stat input file: " + path);
  }}
  const auto expected_bytes =
      expected_elems * static_cast<int64_t>(sizeof(float));
  if (size != static_cast<std::streamoff>(expected_bytes)) {{
    std::ostringstream os;
    os << "unexpected input size for " << path << ": got " << size << " bytes, expected "
       << expected_bytes << " bytes";
    throw std::runtime_error(os.str());
  }}
  std::vector<float> data(static_cast<size_t>(expected_elems));
  in.read(reinterpret_cast<char*>(data.data()), size);
  if (!in) {{
    throw std::runtime_error("failed to read input file: " + path);
  }}
  return data;
}}

bool ReadExact(std::istream& in, void* dst, size_t nbytes) {{
  char* out = static_cast<char*>(dst);
  size_t total = 0;
  while (total < nbytes) {{
    in.read(out + total, static_cast<std::streamsize>(nbytes - total));
    const size_t got = static_cast<size_t>(in.gcount());
    if (got == 0) {{
      if (total == 0 && in.eof()) {{
        return false;
      }}
      throw std::runtime_error("unexpected EOF while reading streamed input");
    }}
    total += got;
  }}
  return true;
}}

std::string JoinPath(const std::string& lhs, const std::string& rhs) {{
  if (lhs.empty()) {{
    return rhs;
  }}
  if (lhs.back() == '/') {{
    return lhs + rhs;
  }}
  return lhs + "/" + rhs;
}}

void WriteFloat32File(const std::string& path, const std::vector<float>& data) {{
  std::ofstream out(path, std::ios::binary);
  if (!out) {{
    throw std::runtime_error("failed to open output bin: " + path);
  }}
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!out) {{
    throw std::runtime_error("failed to write output bin: " + path);
  }}
}}

void WriteTextFile(const std::string& path, const std::vector<float>& data) {{
  std::ofstream out(path);
  if (!out) {{
    throw std::runtime_error("failed to open output txt: " + path);
  }}
  out << std::setprecision(9);
  for (size_t i = 0; i < data.size(); ++i) {{
    out << i << '\\t' << data[i] << '\\n';
  }}
  if (!out) {{
    throw std::runtime_error("failed to write output txt: " + path);
  }}
}}

std::vector<int> TopKIndices(const std::vector<float>& data, size_t k) {{
  std::vector<int> indices(data.size());
  for (size_t i = 0; i < data.size(); ++i) {{
    indices[i] = static_cast<int>(i);
  }}
  if (k > indices.size()) {{
    k = indices.size();
  }}
  std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k),
                    indices.end(), [&data](int lhs, int rhs) {{
                      return data[static_cast<size_t>(lhs)] >
                             data[static_cast<size_t>(rhs)];
                    }});
  indices.resize(k);
  return indices;
}}

void WriteSummaryJson(const std::string& path, const Args& args, double tvm_seconds,
                      const std::vector<float>& output) {{
  std::ofstream out(path);
  if (!out) {{
    throw std::runtime_error("failed to open summary json: " + path);
  }}
  const auto top3 = TopKIndices(output, 3);
  const int pred = static_cast<int>(
      std::distance(output.begin(), std::max_element(output.begin(), output.end())));

  out << "{{\\n";
  out << "  \\"input_bin\\": " << std::quoted(args.input_bin) << ",\\n";
  out << "  \\"library_path\\": " << std::quoted(args.library_path) << ",\\n";
  out << "  \\"constants_dir\\": " << std::quoted(args.constants_dir) << ",\\n";
  out << "  \\"tvm_seconds\\": " << std::fixed << std::setprecision(6) << tvm_seconds << ",\\n";
  out << "  \\"tvm\\": {{\\n";
  out << "    \\"name\\": \\"tvm_aot\\",\\n";
  out << "    \\"shape\\": [{shape_json(output_spec.shape)}],\\n";
  out << "    \\"pred\\": " << pred << ",\\n";
  out << "    \\"top3_indices\\": [";
  for (size_t i = 0; i < top3.size(); ++i) {{
    out << top3[i];
    if (i + 1 < top3.size()) {{
      out << ", ";
    }}
  }}
  out << "],\\n";
  out << "    \\"top3_logits\\": [";
  for (size_t i = 0; i < top3.size(); ++i) {{
    out << output[static_cast<size_t>(top3[i])];
    if (i + 1 < top3.size()) {{
      out << ", ";
    }}
  }}
  out << "],\\n";
  float sum = 0.0f;
  float min_v = output.front();
  float max_v = output.front();
  for (float value : output) {{
    sum += value;
    min_v = std::min(min_v, value);
    max_v = std::max(max_v, value);
  }}
  out << "    \\"logits_sum\\": " << sum << ",\\n";
  out << "    \\"logits_min\\": " << min_v << ",\\n";
  out << "    \\"logits_max\\": " << max_v << "\\n";
  out << "  }}\\n";
  out << "}}\\n";
  if (!out) {{
    throw std::runtime_error("failed to write summary json: " + path);
  }}
}}

Function RequireFunction(const Module& mod, const std::string& name) {{
  auto opt = mod->GetFunction(name, false);
  if (!opt.has_value()) {{
    throw std::runtime_error("missing module function: " + name);
  }}
  return *opt;
}}

Tensor EmptyTensor(std::initializer_list<int64_t> shape) {{
  return Tensor::Empty(tvm::ffi::Shape(shape), DataType(DataType::kFloat, 32, 1),
                       DLDevice{{kDLCPU, 0}});
}}

Tensor TensorFromFile(const std::string& path, int64_t expected_elems,
                      std::initializer_list<int64_t> shape) {{
  const std::vector<float> data = ReadFloat32File(path, expected_elems);
  Tensor tensor = EmptyTensor(shape);
  tensor.CopyFromBytes(data.data(), data.size() * sizeof(float));
  return tensor;
}}

Tensor InputTensorFromFile(const std::string& path, int64_t expected_elems,
                           std::initializer_list<int64_t> shape) {{
  const std::vector<float> data = ReadFloat32File(path, expected_elems);
  Tensor tensor = EmptyTensor(shape);
  const size_t nbytes = data.size() * sizeof(float);
  if (!tvm::runtime::inputzerodither::TryCopyFromBytes(
          data.data(), nbytes, const_cast<DLTensor*>(tensor.operator->()))) {{
    tensor.CopyFromBytes(data.data(), nbytes);
  }}
  return tensor;
}}

Tensor InputTensorFromBytes(const float* data, int64_t expected_elems,
                            std::initializer_list<int64_t> shape) {{
  Tensor tensor = EmptyTensor(shape);
  const size_t nbytes =
      static_cast<size_t>(expected_elems) * sizeof(float);
  if (!tvm::runtime::inputzerodither::TryCopyFromBytes(
          data, nbytes, const_cast<DLTensor*>(tensor.operator->()))) {{
    tensor.CopyFromBytes(data, nbytes);
  }}
  return tensor;
}}

}}  // namespace

int main(int argc, char** argv) {{
  try {{
    const Args args = ParseArgs(argc, argv);

    Module kernels = Module::LoadFromFile(args.library_path);
{chr(10).join(kernel_decl_lines)}
{chr(10).join(const_lines)}
{chr(10).join(tensor_alloc_lines)}
    if (args.stream_mode) {{
      std::vector<float> input(static_cast<size_t>(kInputElems));
      std::vector<float> output(static_cast<size_t>(kOutputElems));
      while (ReadExact(std::cin, input.data(), input.size() * sizeof(float))) {{
        Tensor {input_name} =
            InputTensorFromBytes(input.data(), kInputElems, {shape_literal(input_spec.shape)});
        tvm::runtime::relulow12::ResetInferencePatchSeed();
{chr(10).join(call_lines)}
        {output_storage_name}.CopyToBytes(output.data(), output.size() * sizeof(float));
        std::cout.write(reinterpret_cast<const char*>(output.data()),
                        static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!std::cout) {{
          throw std::runtime_error("failed to write streamed output");
        }}
        std::cout.flush();
      }}
      return 0;
    }}

    Tensor {input_name} =
        InputTensorFromFile(args.input_bin, kInputElems, {shape_literal(input_spec.shape)});
    const auto start = std::chrono::steady_clock::now();
    tvm::runtime::relulow12::ResetInferencePatchSeed();
{chr(10).join(call_lines)}
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::vector<float> output(static_cast<size_t>(kOutputElems));
    {output_storage_name}.CopyToBytes(output.data(), output.size() * sizeof(float));

    WriteFloat32File(args.output_bin, output);
    if (!args.output_txt.empty()) {{
      WriteTextFile(args.output_txt, output);
    }}
    if (!args.summary_json.empty()) {{
      WriteSummaryJson(args.summary_json, args, elapsed.count(), output);
    }}
    return 0;
  }} catch (const std::exception& ex) {{
    std::cerr << "error: " << ex.what() << '\\n';
    return 1;
  }}
}}
"""


def onnx_main_output_spec(model: onnx.ModelProto) -> TensorSpec | None:
    if len(model.graph.output) != 1:
        return None
    output = model.graph.output[0]
    dims: list[int] = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            return None
    dtype = output.type.tensor_type.elem_type
    dtype_name = onnx.TensorProto.DataType.Name(dtype).lower()
    if dtype_name == "float":
        dtype_name = "float32"
    return TensorSpec(shape=tuple(dims), dtype=dtype_name)


def build_aot_kernel_library(
    *,
    onnx_path: Path,
    out_dir: Path,
    library_filename: str,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    tvm, relax, from_onnx, *_ = load_tvm_modules()

    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    constants_dir = build_dir / "constants"
    library_path = build_dir / library_filename
    runner_source_path = build_dir / "generated_runner.cc"

    model = onnx.load(str(onnx_path))
    imported_mod = from_onnx(model)
    inference_mod = relax.transform.DecomposeOpsForInference()(imported_mod)
    lowered_mod = relax.transform.LegalizeOps()(inference_mod)
    kernel_mod, kernel_names = extract_kernel_module(lowered_mod, tvm)

    input_name, input_spec, output_name, output_spec, constants, calls = collect_runner_plan(lowered_mod)
    if input_spec.dtype != "float32" or output_spec.dtype != "float32":
        raise TypeError("only float32 input/output are supported")

    write_text(build_dir / "imported_relax.py", module_to_text(imported_mod))
    write_text(build_dir / "inference_relax.py", module_to_text(inference_mod))
    write_text(build_dir / "lowered_relax.py", module_to_text(lowered_mod))
    write_text(build_dir / "lowered_relax_meta.py", module_to_text(lowered_mod, show_meta=True))
    write_text(build_dir / "tir_kernels.py", module_to_text(kernel_mod))

    constants_manifest = export_constant_blobs(lowered_mod=lowered_mod, constants_dir=constants_dir)
    if len(constants_manifest) != len(constants):
        raise RuntimeError(
            f"constant count mismatch: plan has {len(constants)}, exported {len(constants_manifest)}"
        )

    runner_source = generate_runner_source(
        input_name=input_name,
        input_spec=input_spec,
        output_name=output_name,
        output_spec=output_spec,
        constants=constants,
        calls=calls,
        library_filename=library_filename,
    )
    write_text(runner_source_path, runner_source)

    compile_start = time.perf_counter()
    with tvm.transform.PassContext(opt_level=OPT_LEVEL):
        library = tvm.tirx.build(kernel_mod, target=TARGET)
    compile_seconds = time.perf_counter() - compile_start
    library.export_library(str(library_path))

    opset = 1
    for opset_identifier in model.opset_import:
        if str(opset_identifier.domain) in ("", "ai.onnx"):
            opset = int(opset_identifier.version)
            break

    runner_plan = {
        "input_name": input_name,
        "input_shape": list(input_spec.shape),
        "input_dtype": input_spec.dtype,
        "output_name": output_name,
        "output_shape": list(output_spec.shape),
        "output_dtype": output_spec.dtype,
        "constants": constants_manifest,
        "calls": [
            {
                "output_name": call.output_name,
                "kernel_name": call.kernel_name,
                "inputs": [{"kind": ref.kind, "name": ref.name} for ref in call.inputs],
                "output_shape": list(call.output_spec.shape),
                "output_dtype": call.output_spec.dtype,
            }
            for call in calls
        ],
    }
    write_json(build_dir / "runner_plan.json", runner_plan)

    build_meta = {
        "tvm_home": str(TVM_HOME),
        "tvm_build_dir": str(TVM_BUILD_DIR),
        "tvm_version": str(tvm.__version__),
        "onnx_model": str(onnx_path),
        "library_path": str(library_path),
        "constants_dir": str(constants_dir),
        "generated_runner_source": str(runner_source_path),
        "target": TARGET,
        "opt_level": int(OPT_LEVEL),
        "compile_seconds": float(compile_seconds),
        "onnx_ir_version": int(model.ir_version),
        "onnx_opset": int(opset),
        "kernel_names": kernel_names,
        "num_constants": int(len(constants_manifest)),
        "num_calls": int(len(calls)),
        "input_shape": list(input_spec.shape),
        "output_shape": list(output_spec.shape),
    }
    write_json(build_dir / "build_meta.json", build_meta)
    write_json(build_dir / "constants_manifest.json", {"constants": constants_manifest})
    return library_path, constants_dir, runner_source_path, build_meta


def compile_onnx_to_vm_library(
    *,
    onnx_path: Path,
    out_dir: Path,
    library_filename: str,
) -> tuple[Path, dict[str, Any]]:
    tvm, relax, from_onnx, *_ = load_tvm_modules()

    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / library_filename

    model = onnx.load(str(onnx_path))
    imported_mod = from_onnx(model)
    inference_mod = relax.transform.DecomposeOpsForInference()(imported_mod)
    lowered_mod = relax.transform.LegalizeOps()(inference_mod)

    main = lowered_mod["main"]
    if len(main.params) != 1:
        raise ValueError(f"expected exactly one input parameter, got {len(main.params)}")
    input_spec = tensor_spec_from_sinfo(main.params[0].struct_info)
    output_spec = onnx_main_output_spec(model)

    write_text(build_dir / "imported_relax.py", module_to_text(imported_mod))
    write_text(build_dir / "inference_relax.py", module_to_text(inference_mod))
    write_text(build_dir / "lowered_relax.py", module_to_text(lowered_mod))

    compile_start = time.perf_counter()
    with tvm.transform.PassContext(opt_level=OPT_LEVEL):
        executable = tvm.compile(lowered_mod, target=TARGET)
    compile_seconds = time.perf_counter() - compile_start

    executable.export_library(str(library_path))

    if hasattr(executable, "stats"):
        write_text(build_dir / "vm_stats.txt", executable.stats())
    if hasattr(executable, "as_text"):
        write_text(build_dir / "vm_exec.txt", executable.as_text())

    opset = 1
    for opset_identifier in model.opset_import:
        if str(opset_identifier.domain) in ("", "ai.onnx"):
            opset = int(opset_identifier.version)
            break

    build_meta = {
        "tvm_home": str(TVM_HOME),
        "tvm_build_dir": str(TVM_BUILD_DIR),
        "tvm_version": str(tvm.__version__),
        "onnx_model": str(onnx_path),
        "library_path": str(library_path),
        "target": TARGET,
        "opt_level": int(OPT_LEVEL),
        "compile_seconds": float(compile_seconds),
        "onnx_ir_version": int(model.ir_version),
        "onnx_opset": int(opset),
        "input_shape": list(input_spec.shape),
        "output_shape": list(output_spec.shape) if output_spec is not None else None,
    }
    write_json(build_dir / "build_meta.json", build_meta)
    return library_path, build_meta


def capture_callable(log_path: Path, env_updates: dict[str, str], fn, *args, **kwargs):
    ensure_parent(log_path)
    with log_path.open("w", encoding="ascii") as log_file, scoped_relevant_env(env_updates):
        log_file.write(f"cwd={ROOT_DIR}\n")
        for key in sorted(RELEVANT_ENV_KEYS):
            if key in os.environ:
                log_file.write(f"{key}={os.environ[key]}\n")
        log_file.write(f"begin={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log_file.flush()
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            result = fn(*args, **kwargs)
        log_file.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log_file.flush()
        return result


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    run_env = clean_env()
    if env:
        run_env.update(env)
    ensure_parent(log_path)
    with log_path.open("w", encoding="ascii") as log_file:
        log_file.write(f"cwd={ROOT_DIR}\n")
        log_file.write(f"cmd={render_cmd(cmd)}\n")
        for key in sorted(RELEVANT_ENV_KEYS):
            if key in run_env:
                log_file.write(f"{key}={run_env[key]}\n")
        log_file.write(f"begin={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log_file.flush()
        subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=run_env,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")


def preprocess_image(image: Image.Image, cfg: InputConfig) -> np.ndarray:
    image = image.convert(cfg.source_mode)
    if cfg.resize_hw is not None:
        height, width = cfg.resize_hw
        image = image.resize((width, height), Image.BILINEAR)
    if cfg.grayscale_to_rgb:
        image = image.convert("RGB")

    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    nchw = np.transpose(array, (2, 0, 1))[None, ...]

    mean = np.asarray(cfg.mean, dtype=np.float32).reshape(1, len(cfg.mean), 1, 1)
    std = np.asarray(cfg.std, dtype=np.float32).reshape(1, len(cfg.std), 1, 1)
    if nchw.shape[1] != mean.shape[1]:
        raise ValueError(
            f"channel mismatch during preprocessing: tensor has {nchw.shape[1]} channels, "
            f"mean/std expect {mean.shape[1]}"
        )
    nchw = (nchw - mean) / std
    expected_shape = cfg.input_shape
    if tuple(int(dim) for dim in nchw.shape) != expected_shape:
        raise ValueError(
            f"unexpected preprocessed shape: got {tuple(int(dim) for dim in nchw.shape)}, "
            f"expected {expected_shape}"
        )
    return nchw.astype(np.float32, copy=False)


def load_mnist_raw() -> tuple[np.ndarray, np.ndarray]:
    raw_dir = DATA_ROOT / "mnist" / "MNIST" / "raw"
    images_path = raw_dir / "t10k-images-idx3-ubyte"
    labels_path = raw_dir / "t10k-labels-idx1-ubyte"
    raw = images_path.read_bytes()
    labels_raw = labels_path.read_bytes()
    rows = int.from_bytes(raw[8:12], "big")
    cols = int.from_bytes(raw[12:16], "big")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, rows, cols)
    labels = np.frombuffer(labels_raw, dtype=np.uint8, offset=8)
    return images, labels


def load_cifar_test() -> tuple[np.ndarray, list[int]]:
    batch_path = DATA_ROOT / "cifar10" / "cifar-10-batches-py" / "test_batch"
    with batch_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = np.asarray(payload[b"data"], dtype=np.uint8)
    labels = list(payload[b"labels"])
    return data, labels


def load_imagenet32_val() -> tuple[np.ndarray, list[int]]:
    batch_path = DATA_ROOT / "imagenet" / "val_data"
    with batch_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    data = np.asarray(payload["data"], dtype=np.uint8)
    labels = [int(label) for label in payload["labels"]]
    return data, labels


def load_imagenet50_96_validation() -> tuple[list[Path], list[int], list[str]]:
    label_map_path = IMAGENET50_96_ROOT / "label_map.json"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    class_dirs = [str(value) for value in label_map["class_dirs"]]
    samples: list[Path] = []
    labels: list[int] = []
    for label, class_dir in enumerate(class_dirs):
        class_root = IMAGENET50_96_ROOT / "validation" / class_dir
        for image_path in sorted(path for path in class_root.iterdir() if path.is_file()):
            samples.append(image_path)
            labels.append(label)
    return samples, labels, class_dirs


def load_celeba_test() -> list[tuple[str, int]]:
    partition_path = DATA_ROOT / "celeba" / "annotations" / "list_eval_partition.txt"
    identity_path = DATA_ROOT / "celeba" / "annotations" / "identity_CelebA.txt"
    partitions: dict[str, int] = {}
    for raw in partition_path.read_text(encoding="ascii").splitlines():
        parts = raw.split()
        if len(parts) == 2:
            partitions[parts[0]] = int(parts[1])

    identities: dict[str, int] = {}
    for raw in identity_path.read_text(encoding="ascii").splitlines():
        parts = raw.split()
        if len(parts) == 2:
            identities[parts[0]] = int(parts[1])

    selected: list[tuple[str, int]] = []
    for filename in sorted(identities):
        if partitions.get(filename) != 2:
            continue
        selected.append((filename, identities[filename]))
        if len(selected) >= 2:
            break
    if len(selected) < 2:
        raise RuntimeError("failed to find two CelebA test samples")
    return selected


def load_chest_rows() -> list[dict[str, str]]:
    csv_path = DATA_ROOT / "chestxray14" / "Data_Entry_2017_v2020.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise RuntimeError("failed to find two ChestXray14 rows")
    return rows[:2]


def chest_target_vector(finding_labels: str) -> list[float]:
    target = [0.0] * len(CHEST_LABELS)
    if finding_labels != "No Finding":
        labels = {label.strip() for label in finding_labels.split("|") if label.strip()}
        for index, label in enumerate(CHEST_LABELS):
            if label in labels:
                target[index] = 1.0
    return target


def case_inputs_dir(session_dir: Path, case: BaseCase) -> Path:
    return session_dir / "prepared_inputs" / case.key


def generate_case_inputs(session_dir: Path, case: BaseCase) -> list[InputEntry]:
    out_dir = case_inputs_dir(session_dir, case)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = case.input_cfg

    entries: list[InputEntry] = []
    if cfg.source_family == "mnist":
        images, labels = load_mnist_raw()
        for idx in (0, 1):
            image = Image.fromarray(images[idx], mode="L")
            tensor = preprocess_image(image, cfg)
            bin_path = out_dir / f"sample{idx:04d}_input_nchw_f32.bin"
            meta_path = out_dir / f"sample{idx:04d}.json"
            tensor.tofile(bin_path)
            label = int(labels[idx])
            meta = {
                "sample_index": idx,
                "model_case": case.key,
                "source_family": cfg.source_family,
                "label": label,
                "shape": list(cfg.input_shape),
                "source": f"MNIST:test:{idx}",
                "normalization": {"mean": list(cfg.mean), "std": list(cfg.std)},
                "assumptions": list(cfg.assumptions),
            }
            write_json(meta_path, meta)
            entries.append(
                InputEntry(
                    sample_index=idx,
                    input_bin=str(bin_path),
                    metadata_json=str(meta_path),
                    label=label,
                )
            )
        return entries

    if cfg.source_family == "cifar":
        data, labels = load_cifar_test()
        for idx in (0, 1):
            hwc = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            image = Image.fromarray(hwc, mode="RGB")
            tensor = preprocess_image(image, cfg)
            bin_path = out_dir / f"sample{idx:04d}_input_nchw_f32.bin"
            meta_path = out_dir / f"sample{idx:04d}.json"
            tensor.tofile(bin_path)
            label = int(labels[idx])
            meta = {
                "sample_index": idx,
                "model_case": case.key,
                "source_family": cfg.source_family,
                "label": label,
                "shape": list(cfg.input_shape),
                "source": f"CIFAR10:test:{idx}",
                "normalization": {"mean": list(cfg.mean), "std": list(cfg.std)},
                "assumptions": list(cfg.assumptions),
            }
            write_json(meta_path, meta)
            entries.append(
                InputEntry(
                    sample_index=idx,
                    input_bin=str(bin_path),
                    metadata_json=str(meta_path),
                    label=label,
                )
            )
        return entries

    if cfg.source_family == "imagenet32":
        data, labels = load_imagenet32_val()
        for idx in (0, 1):
            hwc = data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            image = Image.fromarray(hwc, mode="RGB")
            tensor = preprocess_image(image, cfg)
            bin_path = out_dir / f"sample{idx:04d}_input_nchw_f32.bin"
            meta_path = out_dir / f"sample{idx:04d}.json"
            tensor.tofile(bin_path)
            label = int(labels[idx])
            meta = {
                "sample_index": idx,
                "model_case": case.key,
                "source_family": cfg.source_family,
                "label": label,
                "shape": list(cfg.input_shape),
                "source": f"ImageNet32:val:{idx}",
                "normalization": {"mean": list(cfg.mean), "std": list(cfg.std)},
                "assumptions": list(cfg.assumptions),
            }
            write_json(meta_path, meta)
            entries.append(
                InputEntry(
                    sample_index=idx,
                    input_bin=str(bin_path),
                    metadata_json=str(meta_path),
                    label=label,
                )
            )
        return entries

    if cfg.source_family == "imagenet50_96":
        samples, labels, _class_dirs = load_imagenet50_96_validation()
        for idx in (0, 1):
            image_path = samples[idx]
            with Image.open(image_path) as image:
                tensor = preprocess_image(image, cfg)
            bin_path = out_dir / f"sample{idx:04d}_input_nchw_f32.bin"
            meta_path = out_dir / f"sample{idx:04d}.json"
            tensor.tofile(bin_path)
            label = int(labels[idx])
            meta = {
                "sample_index": idx,
                "model_case": case.key,
                "source_family": cfg.source_family,
                "label": label,
                "shape": list(cfg.input_shape),
                "source_image": str(image_path),
                "normalization": {"mean": list(cfg.mean), "std": list(cfg.std)},
                "assumptions": list(cfg.assumptions),
            }
            write_json(meta_path, meta)
            entries.append(
                InputEntry(
                    sample_index=idx,
                    input_bin=str(bin_path),
                    metadata_json=str(meta_path),
                    label=label,
                )
            )
        return entries

    if cfg.source_family == "celea":
        image_root = DATA_ROOT / "celeba" / "Dataset" / "CelebA_train" / "img_align_celeba"
        selected = load_celeba_test()
        for idx, (filename, identity_id) in enumerate(selected):
            image_path = image_root / filename
            with Image.open(image_path) as image:
                tensor = preprocess_image(image, cfg)
            bin_path = out_dir / f"sample{idx:04d}_input_nchw_f32.bin"
            meta_path = out_dir / f"sample{idx:04d}.json"
            tensor.tofile(bin_path)
            label = int(identity_id - 1)
            meta = {
                "sample_index": idx,
                "model_case": case.key,
                "source_family": cfg.source_family,
                "label": label,
                "identity_id": int(identity_id),
                "shape": list(cfg.input_shape),
                "source_image": str(image_path),
                "normalization": {"mean": list(cfg.mean), "std": list(cfg.std)},
                "assumptions": list(cfg.assumptions),
            }
            write_json(meta_path, meta)
            entries.append(
                InputEntry(
                    sample_index=idx,
                    input_bin=str(bin_path),
                    metadata_json=str(meta_path),
                    label=label,
                )
            )
        return entries

    if cfg.source_family == "chest":
        image_root = DATA_ROOT / "chestxray14" / "images"
        rows = load_chest_rows()
        for idx, row in enumerate(rows):
            filename = str(row["Image Index"])
            image_path = image_root / filename
            with Image.open(image_path) as image:
                tensor = preprocess_image(image, cfg)
            bin_path = out_dir / f"sample{idx:04d}_input_nchw_f32.bin"
            meta_path = out_dir / f"sample{idx:04d}.json"
            tensor.tofile(bin_path)
            findings = str(row["Finding Labels"])
            label_vector = chest_target_vector(findings)
            meta = {
                "sample_index": idx,
                "model_case": case.key,
                "source_family": cfg.source_family,
                "label": label_vector,
                "finding_labels": findings,
                "shape": list(cfg.input_shape),
                "source_image": str(image_path),
                "normalization": {"mean": list(cfg.mean), "std": list(cfg.std)},
                "assumptions": list(cfg.assumptions),
            }
            write_json(meta_path, meta)
            entries.append(
                InputEntry(
                    sample_index=idx,
                    input_bin=str(bin_path),
                    metadata_json=str(meta_path),
                    label=label_vector,
                )
            )
        return entries

    raise ValueError(f"unsupported source family: {cfg.source_family}")


def prepare_effective_onnx(session_dir: Path, case: BaseCase) -> dict[str, Any]:
    out_dir = session_dir / "effective_onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    if case.key != "squeezenet_imagenet32":
        return {
            "source_onnx": str(case.onnx_path),
            "effective_onnx": str(case.onnx_path),
            "preprocess_note": "original_onnx",
        }

    optimized_path = out_dir / "squeezenet1_0_imagenet32_ort_basic.onnx"
    optimization_log = out_dir / "squeezenet1_0_imagenet32_ort_basic.log"
    if not optimized_path.exists():
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = str(optimized_path)
        providers = ["CPUExecutionProvider"]
        with optimization_log.open("w", encoding="ascii") as log_file:
            log_file.write(f"source_onnx={case.onnx_path}\n")
            log_file.write(f"optimized_onnx={optimized_path}\n")
            log_file.write("graph_optimization_level=ORT_ENABLE_BASIC\n")
            log_file.flush()
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                ort.InferenceSession(str(case.onnx_path), so, providers=providers)
            log_file.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")

    return {
        "source_onnx": str(case.onnx_path),
        "effective_onnx": str(optimized_path),
        "preprocess_note": "onnxruntime_basic_optimized_copy_for_tvm_frontend_compat",
        "optimization_log": str(optimization_log),
    }


def build_generic_vm_runner(session_dir: Path) -> Path:
    out_dir = session_dir / "runners" / "common"
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "generic_tvm_native_vm_runner"
    log_path = session_dir / "build_logs" / "common" / "generic_vm_runner.log"
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        f"-I{TVM_HOME / 'include'}",
        f"-I{TVM_HOME / '3rdparty' / 'tvm-ffi' / 'include'}",
        f"-I{TVM_HOME / '3rdparty' / 'tvm-ffi' / '3rdparty' / 'dlpack' / 'include'}",
        str(GENERIC_VM_RUNNER_SRC),
        f"-L{TVM_BUILD_DIR}",
        f"-L{TVM_BUILD_DIR / 'lib'}",
        f"-Wl,-rpath,{TVM_BUILD_DIR}",
        f"-Wl,-rpath,{TVM_BUILD_DIR / 'lib'}",
        "-ltvm_ffi",
        "-ltvm_runtime",
        "-ldl",
        "-lpthread",
        "-o",
        str(bin_path),
    ]
    run_logged(cmd, log_path)
    return bin_path


def file_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for chunk in handle:
            if needle in chunk:
                return True
    return False


def validate_taintblock16trace(trace_json_path: Path, taint_file: str) -> dict[str, Any]:
    checks = {
        "trace_json_exists": trace_json_path.exists(),
        "block_size_16": file_contains(trace_json_path, '"block_size": 16'),
        "taint_seed_mode_file": file_contains(trace_json_path, '"taint_seed_mode": "file"'),
        "taint_file_matches_input": file_contains(trace_json_path, f'"taint_file": "{taint_file}"'),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"taintblock16trace validation failed for {trace_json_path}: {checks}"
        )
    return checks


def compare_outputs(reference_bin: Path, trace_bin: Path, out_json: Path) -> None:
    reference = np.fromfile(reference_bin, dtype=np.float32)
    traced = np.fromfile(trace_bin, dtype=np.float32)
    if reference.shape != traced.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)} vs trace {tuple(traced.shape)}"
        )
    diff = traced - reference
    top_ref = np.argsort(reference)[::-1][:3]
    top_trace = np.argsort(traced)[::-1][:3]
    payload = {
        "reference_npy": str(reference_bin),
        "trace_npy": str(trace_bin),
        "shape": list(reference.shape),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "l1_diff": float(np.sum(np.abs(diff))) if diff.size else 0.0,
        "allclose_rtol_1e-5_atol_1e-6": bool(
            np.allclose(reference, traced, rtol=1e-5, atol=1e-6)
        ),
        "reference_pred": int(np.argmax(reference)) if reference.size else None,
        "reference_top3_indices": [int(i) for i in top_ref],
        "reference_top3_logits": [float(reference[i]) for i in top_ref],
        "trace_pred": int(np.argmax(traced)) if traced.size else None,
        "trace_top3_indices": [int(i) for i in top_trace],
        "trace_top3_logits": [float(traced[i]) for i in top_trace],
    }
    write_json(out_json, payload)


def verify_build(
    *,
    spec: RuntimeSpec,
    mode: str,
    build_dir: Path,
    library_path: Path,
    marker_file: Path,
    constants_dir: Path | None,
) -> dict[str, Any]:
    marker_text = marker_file.read_text(encoding="ascii", errors="ignore")
    nm_result = subprocess.run(
        ["nm", "-D", str(library_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    has_relu_marker = "tvm_relu_low12_f32" in marker_text or "tvm_relu_low12_f32_scalar" in marker_text
    has_relu6_marker = "tvm_relu6_low12_f32" in marker_text or "tvm_relu6_low12_f32_scalar" in marker_text
    has_relu_symbol_ref = "tvm_relu_low12_f32_scalar" in nm_result.stdout
    has_relu6_symbol_ref = "tvm_relu6_low12_f32_scalar" in nm_result.stdout
    has_patch_symbol_ref = has_relu_symbol_ref or has_relu6_symbol_ref
    expected = mode == "on"
    if has_patch_symbol_ref != expected:
        raise RuntimeError(
            f"build verification failed for {spec.spec_id}/{mode}: "
            f"has_relu_marker={has_relu_marker}, has_relu6_marker={has_relu6_marker}, "
            f"has_relu_symbol_ref={has_relu_symbol_ref}, has_relu6_symbol_ref={has_relu6_symbol_ref}"
        )
    return {
        "spec_id": spec.spec_id,
        "mode": mode,
        "build_dir": str(build_dir),
        "library_path": str(library_path),
        "marker_file": str(marker_file),
        "constants_dir": str(constants_dir) if constants_dir is not None else None,
        "has_tvm_relu_low12_f32_in_ir": has_relu_marker,
        "has_tvm_relu6_low12_f32_in_ir": has_relu6_marker,
        "has_tvm_relu_low12_f32_symbol_ref": has_relu_symbol_ref,
        "has_tvm_relu6_low12_f32_symbol_ref": has_relu6_symbol_ref,
    }


def build_library_filename(spec: RuntimeSpec) -> str:
    if spec.runtime == "native_vm":
        return f"{spec.spec_id}_tvm.so"
    return f"{spec.spec_id}_tvm_aot_kernels.so"


def aot_runner_binary_name(spec: RuntimeSpec) -> str:
    return f"{spec.spec_id}_tvm_aot_runner"


def build_mode_main(session_dir: Path, spec_id: str, mode: str) -> int:
    spec = runtime_specs()[spec_id]
    manifest = worker_manifest(session_dir)
    effective_onnx = Path(manifest["effective_onnx"][spec.base.key]["effective_onnx"])
    library_filename = build_library_filename(spec)
    out_dir = session_dir / "artifacts" / spec.spec_id / mode
    log_path = session_dir / "build_logs" / spec.spec_id / f"{mode}.log"
    env_updates = {"TVM_RELU_LOW12_PATCH": "1"} if mode == "on" else {}

    if spec.runtime == "native_vm":
        library_path, build_meta = capture_callable(
            log_path,
            env_updates,
            compile_onnx_to_vm_library,
            onnx_path=effective_onnx,
            out_dir=out_dir,
            library_filename=library_filename,
        )
        payload = {
            "spec_id": spec.spec_id,
            "mode": mode,
            "runtime": spec.runtime,
            "out_dir": str(out_dir),
            "library_path": str(library_path),
            "constants_dir": None,
            "runner_source_path": None,
            "build_meta": build_meta,
        }
    else:
        library_path, constants_dir, runner_source_path, build_meta = capture_callable(
            log_path,
            env_updates,
            build_aot_kernel_library,
            onnx_path=effective_onnx,
            out_dir=out_dir,
            library_filename=library_filename,
        )
        payload = {
            "spec_id": spec.spec_id,
            "mode": mode,
            "runtime": spec.runtime,
            "out_dir": str(out_dir),
            "library_path": str(library_path),
            "constants_dir": str(constants_dir),
            "runner_source_path": str(runner_source_path),
            "build_meta": build_meta,
        }

    write_json(session_dir / "build_status" / spec.spec_id / f"{mode}.json", payload)
    return 0


def build_spec_artifacts(session_dir: Path, spec: RuntimeSpec, vm_runner_bin: Path) -> dict[str, Any]:
    build_root = session_dir / "artifacts" / spec.spec_id
    build_logs_root = session_dir / "build_logs" / spec.spec_id
    build_logs_root.mkdir(parents=True, exist_ok=True)

    build_reports: dict[str, Any] = {}
    build_verification: dict[str, Any] = {}

    for mode in ("off", "on"):
        build_cmd = [
            str(PYTHON_BIN),
            str(FULL_MATRIX_DIR / "run_matrix.py"),
            "build-mode",
            "--session-dir",
            str(session_dir),
            "--spec",
            spec.spec_id,
            "--mode",
            mode,
        ]
        build_env = clean_env()
        if mode == "on":
            build_env["TVM_RELU_LOW12_PATCH"] = "1"
        subprocess.run(build_cmd, cwd=ROOT_DIR, env=build_env, check=True)

        build_status = read_json(session_dir / "build_status" / spec.spec_id / f"{mode}.json")
        out_dir = Path(build_status["out_dir"])
        library_path = Path(build_status["library_path"])
        build_meta = build_status["build_meta"]
        constants_dir = (
            Path(build_status["constants_dir"])
            if build_status["constants_dir"] is not None
            else None
        )
        if spec.runtime == "native_vm":
            marker_file = out_dir / "build" / "lowered_relax.py"
        else:
            marker_file = out_dir / "build" / "tir_kernels.py"

        build_reports[mode] = {
            "out_dir": str(out_dir),
            "build_meta": build_meta,
        }
        build_verification[mode] = verify_build(
            spec=spec,
            mode=mode,
            build_dir=out_dir / "build",
            library_path=Path(library_path),
            marker_file=marker_file,
            constants_dir=constants_dir if spec.runtime == "aot" else None,
        )

    runner_info: dict[str, Any]
    if spec.runtime == "native_vm":
        runner_info = {
            "runner_bin": str(vm_runner_bin),
            "input_shape": list(spec.base.input_cfg.input_shape),
        }
    else:
        runner_bin = session_dir / "runners" / spec.spec_id / aot_runner_binary_name(spec)
        runner_bin.parent.mkdir(parents=True, exist_ok=True)
        runner_source = Path(build_reports["on"]["build_meta"]["generated_runner_source"])
        runner_log = build_logs_root / "runner.log"
        cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{TVM_HOME / 'include'}",
            f"-I{TVM_HOME / '3rdparty' / 'tvm-ffi' / 'include'}",
            f"-I{TVM_HOME / '3rdparty' / 'tvm-ffi' / '3rdparty' / 'dlpack' / 'include'}",
            str(runner_source),
            f"-L{TVM_BUILD_DIR}",
            f"-L{TVM_BUILD_DIR / 'lib'}",
            f"-Wl,-rpath,{TVM_BUILD_DIR}",
            f"-Wl,-rpath,{TVM_BUILD_DIR / 'lib'}",
            "-ltvm_ffi",
            "-ltvm_runtime",
            "-ldl",
            "-lpthread",
            "-o",
            str(runner_bin),
        ]
        run_logged(cmd, runner_log)
        runner_info = {
            "runner_bin": str(runner_bin),
            "runner_log": str(runner_log),
        }

    return {
        "build_reports": build_reports,
        "build_verification": build_verification,
        "runner_info": runner_info,
    }


def run_reference_and_pin(
    *,
    spec: RuntimeSpec,
    runner_bin: Path,
    library_path: Path,
    input_entry: InputEntry,
    mode: str,
    run_dir: Path,
    input_shape: tuple[int, int, int, int],
    constants_dir: Path | None,
) -> None:
    env = mode_env(mode)
    run_dir.mkdir(parents=True, exist_ok=True)

    ref_bin = run_dir / "reference_output.bin"
    ref_json = run_dir / "reference_summary.json"
    ref_txt = run_dir / "reference_output.txt"
    ref_stdout = run_dir / "reference.stdout.txt"
    ref_stderr = run_dir / "reference.stderr.txt"
    trace_out_bin = run_dir / "trace_output.bin"
    trace_summary_json = run_dir / "trace_summary.json"
    trace_txt = run_dir / "trace_output.txt"
    pin_stdout = run_dir / "pin.stdout.txt"
    pin_stderr = run_dir / "pin.stderr.txt"
    taint_block_json = run_dir / "taint_block16_bits.json"
    taintblock_ip = run_dir / "taintblock16trace.ip.txt"
    verify_json = run_dir / "verify.json"
    execution_proof = run_dir / "execution_proof.json"
    run_log = run_dir / "run.log"
    cmd_txt = run_dir / "cmd.txt"

    if spec.runtime == "native_vm":
        shape_text = ",".join(str(dim) for dim in input_shape)
        reference_cmd = [
            str(runner_bin),
            "--library",
            str(library_path),
            "--input-bin",
            input_entry.input_bin,
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
            input_entry.input_bin,
            "--input-shape",
            shape_text,
            "--output-bin",
            str(trace_out_bin),
            "--summary-json",
            str(trace_summary_json),
            "--output-txt",
            str(trace_txt),
        ]
    else:
        if constants_dir is None:
            raise ValueError("constants_dir is required for AOT runs")
        reference_cmd = [
            str(runner_bin),
            "--library",
            str(library_path),
            "--constants-dir",
            str(constants_dir),
            "--input-bin",
            input_entry.input_bin,
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
            input_entry.input_bin,
            "--output-bin",
            str(trace_out_bin),
            "--summary-json",
            str(trace_summary_json),
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
        input_entry.input_bin,
        "-o",
        str(taint_block_json),
        "-m",
        str(taintblock_ip),
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
                f"spec_id={spec.spec_id}",
                f"mode={mode}",
                f"library_path={library_path}",
                f"runner_bin={runner_bin}",
                f"input_bin={input_entry.input_bin}",
                f"constants_dir={constants_dir if constants_dir is not None else ''}",
                f"env_TVM_RELU_LOW12_PATCH={env.get('TVM_RELU_LOW12_PATCH', '')}",
                f"env_TVM_RELU_PATCH_BITS={env.get('TVM_RELU_PATCH_BITS', '')}",
                f"env_TVM_RELU_PATCH_FIXED_BITS={env.get('TVM_RELU_PATCH_FIXED_BITS', '')}",
                f"env_TVM_RELU_PATCH_FIXED_VALUE={env.get('TVM_RELU_PATCH_FIXED_VALUE', '')}",
                f"env_TVM_RELU_PATCH_INC={env.get('TVM_RELU_PATCH_INC', '')}",
                f"env_TVM_RELU_PATCH_POSITIVE={env.get('TVM_RELU_PATCH_POSITIVE', '')}",
                f"env_TVM_INPUT_ZERO_DITHER={env.get('TVM_INPUT_ZERO_DITHER', '')}",
                f"env_TVM_INPUT_ZERO_DITHER_LAYOUT={env.get('TVM_INPUT_ZERO_DITHER_LAYOUT', '')}",
                f"env_TVM_INPUT_ZERO_DITHER_THRESH={env.get('TVM_INPUT_ZERO_DITHER_THRESH', '')}",
                f"env_TVM_INPUT_ZERO_DITHER_EPS_MIN={env.get('TVM_INPUT_ZERO_DITHER_EPS_MIN', '')}",
                f"env_TVM_INPUT_ZERO_DITHER_EPS_MAX={env.get('TVM_INPUT_ZERO_DITHER_EPS_MAX', '')}",
                f"disable_aslr={'1' if wrapped_reference_cmd != reference_cmd else '0'}",
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
            cwd=ROOT_DIR,
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
            cwd=ROOT_DIR,
            env={**clean_env(), **env},
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    compare_outputs(ref_bin, trace_out_bin, verify_json)
    trace_validation = validate_taintblock16trace(taint_block_json, input_entry.input_bin)

    saw_relu_helper = file_contains(taintblock_ip, "tvm_relu_low12_f32") or file_contains(
        taintblock_ip, "tvm_relu6_low12_f32"
    )
    saw_input_dither = file_contains(ref_stderr, "input_zero_dither=random") or file_contains(
        pin_stderr, "input_zero_dither=random"
    )
    expected = mode == "on"
    if saw_relu_helper != expected:
        raise RuntimeError(
            f"relu/relu6 helper execution proof mismatch for {spec.spec_id}/{mode}: "
            f"{saw_relu_helper}"
        )
    if saw_input_dither != expected:
        raise RuntimeError(
            f"input dither execution proof mismatch for {spec.spec_id}/{mode}: {saw_input_dither}"
        )

    write_json(
        execution_proof,
        {
            "spec_id": spec.spec_id,
            "mode": mode,
            "run_dir": str(run_dir),
            "run_log": str(run_log),
            "verify_json": str(verify_json),
            "trace_kind": "taintblock16trace",
            "trace_json": str(taint_block_json),
            "trace_ipmap": str(taintblock_ip),
            "trace_validation": trace_validation,
            "relu_helper_seen_in_ipmap": saw_relu_helper,
            "input_dither_logged": saw_input_dither,
            "requested_config": REQUESTED_CONFIG,
        },
    )
    with run_log.open("a", encoding="ascii") as handle:
        handle.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")


def worker_manifest(session_dir: Path) -> dict[str, Any]:
    return read_json(session_dir / "manifest.json")


def worker_main(session_dir: Path, spec_id: str) -> int:
    specs = runtime_specs()
    if spec_id not in specs:
        raise ValueError(f"unknown spec: {spec_id}")
    spec = specs[spec_id]
    manifest = worker_manifest(session_dir)
    worker_log = session_dir / "worker_logs" / f"{spec.spec_id}.log"
    ensure_parent(worker_log)

    with worker_log.open("w", encoding="ascii") as log_file, contextlib.redirect_stdout(
        log_file
    ), contextlib.redirect_stderr(log_file):
        print(f"spec_id={spec.spec_id}")
        print(f"runtime={spec.runtime}")
        print(f"base_case={spec.base.key}")
        print(f"begin={time.strftime('%Y-%m-%dT%H:%M:%S%z')}")

        vm_runner_bin = Path(manifest["common_artifacts"]["generic_vm_runner"])
        build_bundle = build_spec_artifacts(session_dir, spec, vm_runner_bin)

        entries = [InputEntry(**entry) for entry in manifest["inputs"][spec.base.key]]
        runner_bin = Path(build_bundle["runner_info"]["runner_bin"])
        status_runs: list[dict[str, Any]] = []

        for entry in entries:
            for mode in ("off", "on"):
                build_dir = session_dir / "artifacts" / spec.spec_id / mode / "build"
                library_path = build_dir / build_library_filename(spec)
                constants_dir = build_dir / "constants" if spec.runtime == "aot" else None
                out_root = session_dir / "runs" / spec.spec_id / f"sample{entry.sample_index:04d}"
                run_dir = out_root / mode
                run_reference_and_pin(
                    spec=spec,
                    runner_bin=runner_bin,
                    library_path=library_path,
                    input_entry=entry,
                    mode=mode,
                    run_dir=run_dir,
                    input_shape=spec.base.input_cfg.input_shape,
                    constants_dir=constants_dir,
                )
                proof = read_json(run_dir / "execution_proof.json")
                proof["sample_index"] = entry.sample_index
                proof["label"] = entry.label
                status_runs.append(proof)
                write_json(
                    session_dir
                    / "run_status"
                    / spec.spec_id
                    / f"sample{entry.sample_index:04d}_{mode}.json",
                    proof,
                )

        payload = {
            "spec_id": spec.spec_id,
            "runtime": spec.runtime,
            "base_case": spec.base.key,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_reports": build_bundle["build_reports"],
            "build_verification": build_bundle["build_verification"],
            "runner_info": build_bundle["runner_info"],
            "runs": status_runs,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(session_dir / "worker_status" / f"{spec.spec_id}.json", payload)
        print(f"finished={payload['finished_at']}")
    return 0


def prepare_manifest(session_dir: Path, spec_ids: list[str], parallelism: int) -> dict[str, Any]:
    spec_map = runtime_specs()
    selected_specs = [spec_map[spec_id] for spec_id in spec_ids]
    selected_case_keys = []
    for spec in selected_specs:
        if spec.base.key not in selected_case_keys:
            selected_case_keys.append(spec.base.key)

    cases = base_case_map()
    inputs = {case_key: [asdict(entry) for entry in generate_case_inputs(session_dir, cases[case_key])] for case_key in selected_case_keys}
    effective_onnx = {
        case_key: prepare_effective_onnx(session_dir, cases[case_key]) for case_key in selected_case_keys
    }
    generic_vm_runner = build_generic_vm_runner(session_dir)

    manifest = {
        "session_dir": str(session_dir),
        "root_dir": str(ROOT_DIR),
        "full_matrix_dir": str(FULL_MATRIX_DIR),
        "glow_root": str(GLOW_ROOT),
        "python_bin": str(PYTHON_BIN),
        "requested_config": REQUESTED_CONFIG,
        "parallelism": int(parallelism),
        "selected_spec_ids": spec_ids,
        "base_cases": {case.key: base_case_to_json(case) for case in BASE_CASES if case.key in selected_case_keys},
        "specs": {spec_id: spec_to_json(spec_map[spec_id]) for spec_id in spec_ids},
        "inputs": inputs,
        "effective_onnx": effective_onnx,
        "common_artifacts": {
            "generic_vm_runner": str(generic_vm_runner),
        },
    }
    write_json(session_dir / "manifest.json", manifest)
    return manifest


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
                str(FULL_MATRIX_DIR / "run_matrix.py"),
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


def list_main(spec_ids: list[str]) -> int:
    specs = runtime_specs()
    for spec_id in spec_ids:
        spec = specs[spec_id]
        print(
            f"{spec.spec_id}\t{spec.runtime}\t{spec.base.model_family}\t{spec.base.dataset}\t"
            f"{spec.base.metric}\t{spec.base.onnx_path}"
        )
    return 0


def launch_main(session_dir: Path, spec_ids: list[str], parallelism: int) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    manager_log = session_dir / "manager.log"
    cmd = [
        str(PYTHON_BIN),
        str(FULL_MATRIX_DIR / "run_matrix.py"),
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
            "requested_config": REQUESTED_CONFIG,
            "notes": [
                "TV01-TV17 are native_vm cases; TA01-TA17 are aot cases.",
                "TVM_RELU_PATCH_FIXED_VALUE uses 0x75 because the runtime parser does not accept 0b1110101.",
                "SqueezeNet imagenet32 uses a session-local ORT basic optimized ONNX copy for TVM frontend compatibility.",
            ],
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
            "state": "preparing",
            "selected_spec_ids": spec_ids,
            "parallelism": int(parallelism),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    manifest = prepare_manifest(session_dir, spec_ids, parallelism)
    write_json(
        manager_status_path,
        {
            "session_dir": str(session_dir),
            "state": "running",
            "selected_spec_ids": spec_ids,
            "parallelism": int(parallelism),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "manifest": str(session_dir / "manifest.json"),
            "generic_vm_runner": manifest["common_artifacts"]["generic_vm_runner"],
        },
    )

    completed = launch_workers(session_dir, spec_ids, parallelism)
    failures = sorted([spec_id for spec_id, result in completed.items() if result["returncode"] != 0])
    payload = {
        "session_dir": str(session_dir),
        "state": "finished" if not failures else "finished_with_failures",
        "selected_spec_ids": spec_ids,
        "parallelism": int(parallelism),
        "manifest": str(session_dir / "manifest.json"),
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

    spec_ids = selected_spec_ids(getattr(args, "specs", None))

    if args.command == "list":
        return list_main(spec_ids)
    if args.command == "launch":
        return launch_main(make_session_dir(args.session_dir), spec_ids, args.parallelism)
    if args.command == "manager":
        return manager_main(Path(args.session_dir).resolve(), spec_ids, args.parallelism)
    if args.command == "build-mode":
        return build_mode_main(Path(args.session_dir).resolve(), args.spec, args.mode)
    if args.command == "worker":
        return worker_main(Path(args.session_dir).resolve(), args.spec)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
