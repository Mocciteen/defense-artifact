from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path("/path/to/tvm_ana_workspace")
FULL_MATRIX_DIR = ROOT_DIR / "full_matrix"
SESSION_PARENT = FULL_MATRIX_DIR / "sessions"
GLOW_ROOT = Path("/path/to/glow_workspace")
DATA_ROOT = Path("/path/to/datasets")
IMAGENET50_96_ROOT = Path("/path/to/datasets/imagenet50_96")
PYTHON_BIN = ROOT_DIR / ".venv_ciphersteal" / "bin" / "python"
PIN_BIN = GLOW_ROOT / "pin" / "pin"
PINTOOL = (
    GLOW_ROOT
    / "build_release_all"
    / "glow"
    / "pintrace"
    / "obj-intel64"
    / "taintblock16trace.so"
)
GENERIC_VM_RUNNER_SRC = FULL_MATRIX_DIR / "generic_vm_runner.cc"
TVM_BUILD_DIR = ROOT_DIR / "tvm" / "build-llvm18"
TVM_HOME = ROOT_DIR / "tvm"


RELU_PATCH_ENV_ON = {
    "TVM_RELU_LOW12_PATCH": "1",
    "TVM_RELU_PATCH_BITS": "30",
    "TVM_RELU_PATCH_FIXED_BITS": "7",
    # strtoul(..., base=0) accepts 0x75, not 0b1110101.
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VGG_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
VGG_CIFAR_STD = (0.2023, 0.1994, 0.2010)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DENSENET_RGB_HALF = (0.5, 0.5, 0.5)
DENSENET_GRAY_HALF = (0.5,)


CHEST_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)


@dataclass(frozen=True)
class InputConfig:
    source_family: str
    input_shape: tuple[int, int, int, int]
    resize_hw: tuple[int, int] | None
    source_mode: str
    grayscale_to_rgb: bool
    mean: tuple[float, ...]
    std: tuple[float, ...]
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaseCase:
    ordinal: int
    key: str
    model_family: str
    dataset: str
    task_kind: str
    metric: str
    onnx_path: Path
    input_cfg: InputConfig


@dataclass(frozen=True)
class RuntimeSpec:
    spec_id: str
    runtime: str
    runtime_label: str
    base: BaseCase


BASE_CASES: tuple[BaseCase, ...] = (
    BaseCase(
        ordinal=1,
        key="lenet_mnist",
        model_family="lenet",
        dataset="mnist",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "lenet" / "exports" / "mnist" / "lenet_mnist.onnx",
        input_cfg=InputConfig(
            source_family="mnist",
            input_shape=(1, 1, 28, 28),
            resize_hw=None,
            source_mode="L",
            grayscale_to_rgb=False,
            mean=MNIST_MEAN,
            std=MNIST_STD,
        ),
    ),
    BaseCase(
        ordinal=2,
        key="vggnet_mnist",
        model_family="vggnet",
        dataset="mnist",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "vggnet" / "exports" / "mnist" / "vgg_mnist.onnx",
        input_cfg=InputConfig(
            source_family="mnist",
            input_shape=(1, 1, 28, 28),
            resize_hw=None,
            source_mode="L",
            grayscale_to_rgb=False,
            mean=MNIST_MEAN,
            std=MNIST_STD,
        ),
    ),
    BaseCase(
        ordinal=3,
        key="vggnet_cifar",
        model_family="vggnet",
        dataset="cifar",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "vggnet" / "exports" / "cifar" / "vgg_cifar.onnx",
        input_cfg=InputConfig(
            source_family="cifar",
            input_shape=(1, 3, 32, 32),
            resize_hw=None,
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=VGG_CIFAR_MEAN,
            std=VGG_CIFAR_STD,
        ),
    ),
    BaseCase(
        ordinal=4,
        key="squeezenet_cifar",
        model_family="squeezenet",
        dataset="cifar",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "SqueezeNet" / "exports" / "cifar" / "squeezenet1_0_cifar.onnx",
        input_cfg=InputConfig(
            source_family="cifar",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=5,
        key="squeezenet_imagenet50_96",
        model_family="squeezenet",
        dataset="imagenet50_96",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT
        / "model"
        / "SqueezeNet"
        / "exports"
        / "imagenet50_96"
        / "squeezenet1_0_imagenet50_96.onnx",
        input_cfg=InputConfig(
            source_family="imagenet50_96",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=6,
        key="resnet_mnist",
        model_family="resnet",
        dataset="mnist",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "resnet" / "exports" / "mnist" / "resnet18_mnist.onnx",
        input_cfg=InputConfig(
            source_family="mnist",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="L",
            grayscale_to_rgb=True,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=7,
        key="resnet_cifar",
        model_family="resnet",
        dataset="cifar",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "resnet" / "exports" / "cifar" / "resnet18_cifar.onnx",
        input_cfg=InputConfig(
            source_family="cifar",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=8,
        key="resnet_imagenet50_96",
        model_family="resnet",
        dataset="imagenet50_96",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT
        / "model"
        / "resnet"
        / "exports"
        / "imagenet50_96"
        / "resnet18_imagenet50_96.onnx",
        input_cfg=InputConfig(
            source_family="imagenet50_96",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=9,
        key="resnet_celea",
        model_family="resnet",
        dataset="celea",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "resnet" / "exports" / "celea" / "resnet18_celea.onnx",
        input_cfg=InputConfig(
            source_family="celea",
            input_shape=(1, 3, 224, 224),
            resize_hw=(224, 224),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=10,
        key="resnet_chest",
        model_family="resnet",
        dataset="chest",
        task_kind="multi_label",
        metric="macro_auroc",
        onnx_path=GLOW_ROOT / "model" / "resnet" / "exports" / "chest" / "resnet18_chest.onnx",
        input_cfg=InputConfig(
            source_family="chest",
            input_shape=(1, 3, 224, 224),
            resize_hw=(224, 224),
            source_mode="L",
            grayscale_to_rgb=True,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=11,
        key="mobilenet_mnist",
        model_family="mobilenet",
        dataset="mnist",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT
        / "model"
        / "mobilenet"
        / "exports"
        / "mnist"
        / "mobilenet_v2_mnist.onnx",
        input_cfg=InputConfig(
            source_family="mnist",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="L",
            grayscale_to_rgb=True,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=12,
        key="mobilenet_cifar",
        model_family="mobilenet",
        dataset="cifar",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT
        / "model"
        / "mobilenet"
        / "exports"
        / "cifar"
        / "mobilenet_v2_cifar.onnx",
        input_cfg=InputConfig(
            source_family="cifar",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=13,
        key="mobilenet_celea",
        model_family="mobilenet",
        dataset="celea",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT
        / "model"
        / "mobilenet"
        / "exports"
        / "celea"
        / "mobilenet_v2_celea.onnx",
        input_cfg=InputConfig(
            source_family="celea",
            input_shape=(1, 3, 224, 224),
            resize_hw=(224, 224),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=14,
        key="mobilenet_chest",
        model_family="mobilenet",
        dataset="chest",
        task_kind="multi_label",
        metric="macro_auroc",
        onnx_path=GLOW_ROOT
        / "model"
        / "mobilenet"
        / "exports"
        / "chest"
        / "mobilenet_v2_chest.onnx",
        input_cfg=InputConfig(
            source_family="chest",
            input_shape=(1, 3, 224, 224),
            resize_hw=(224, 224),
            source_mode="L",
            grayscale_to_rgb=True,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=15,
        key="densenet_imagenet50_96",
        model_family="densenet",
        dataset="imagenet50_96",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT
        / "model"
        / "densenet"
        / "exports"
        / "imagenet50_96"
        / "densenet121_imagenet50_96.onnx",
        input_cfg=InputConfig(
            source_family="imagenet50_96",
            input_shape=(1, 3, 96, 96),
            resize_hw=(96, 96),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ),
    BaseCase(
        ordinal=16,
        key="densenet_celea",
        model_family="densenet",
        dataset="celea",
        task_kind="single_label",
        metric="accuracy",
        onnx_path=GLOW_ROOT / "model" / "densenet" / "exports" / "celea" / "densenet_celea.onnx",
        input_cfg=InputConfig(
            source_family="celea",
            input_shape=(1, 3, 218, 178),
            resize_hw=(218, 178),
            source_mode="RGB",
            grayscale_to_rgb=False,
            mean=DENSENET_RGB_HALF,
            std=DENSENET_RGB_HALF,
        ),
    ),
    BaseCase(
        ordinal=17,
        key="densenet_chest",
        model_family="densenet",
        dataset="chest",
        task_kind="multi_label",
        metric="macro_auroc",
        onnx_path=GLOW_ROOT / "model" / "densenet" / "exports" / "chest" / "densenet_chest.onnx",
        input_cfg=InputConfig(
            source_family="chest",
            input_shape=(1, 1, 224, 224),
            resize_hw=(224, 224),
            source_mode="L",
            grayscale_to_rgb=False,
            mean=DENSENET_GRAY_HALF,
            std=DENSENET_GRAY_HALF,
        ),
    ),
)


def base_case_map() -> dict[str, BaseCase]:
    return {case.key: case for case in BASE_CASES}


def runtime_specs() -> dict[str, RuntimeSpec]:
    specs: dict[str, RuntimeSpec] = {}
    for case in BASE_CASES:
        tv_id = f"TV{case.ordinal:02d}"
        ta_id = f"TA{case.ordinal:02d}"
        specs[tv_id] = RuntimeSpec(
            spec_id=tv_id,
            runtime="native_vm",
            runtime_label="native VM runtime",
            base=case,
        )
        specs[ta_id] = RuntimeSpec(
            spec_id=ta_id,
            runtime="aot",
            runtime_label="AOT runtime",
            base=case,
        )
    return specs


def ordered_spec_ids() -> list[str]:
    out: list[str] = []
    for case in BASE_CASES:
        out.append(f"TV{case.ordinal:02d}")
    for case in BASE_CASES:
        out.append(f"TA{case.ordinal:02d}")
    return out
