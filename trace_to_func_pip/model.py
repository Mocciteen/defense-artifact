from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


Layer = tuple[torch.Tensor, torch.Tensor]


def prod(values: tuple[int, ...]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


class ConvEncoder(nn.Module):
    def __init__(self, channels: int, size: int, out_dim: int):
        super().__init__()
        if size < 4 or size & (size - 1):
            raise ValueError("conv mixer expects a square power-of-two folded trace")
        width = 64
        depth = int(math.log2(size)) - 1
        layers: list[nn.Module] = [self.block(channels, width)]
        for index in range(max(depth - 2, 0)):
            layers.append(self.block(min(2**index, 8) * width, min(2 ** (index + 1), 8) * width))
        layers.extend([nn.Conv2d(min(2 ** max(depth - 2, 0), 8) * width, out_dim, 4, 1, 0), nn.BatchNorm2d(out_dim)])
        self.net = nn.Sequential(*layers)

    @staticmethod
    def block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, 2, 1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        return self.net(traces).flatten(1)


class Mixer(nn.Module):
    def __init__(self, trace_shape: tuple[int, ...], z_dim: int, n_layers: int, hidden: int, kind: str = "auto"):
        super().__init__()
        self.z_dim = int(z_dim)
        self.n_layers = int(n_layers)
        out_dim = self.z_dim * self.n_layers
        if kind == "auto":
            kind = "conv" if len(trace_shape) == 3 and trace_shape[-1] == trace_shape[-2] else "mlp"
        if kind == "conv":
            channels, height, width = trace_shape
            if height != width:
                raise ValueError("conv mixer expects square folded traces")
            self.net = ConvEncoder(int(channels), int(height), out_dim)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(prod(trace_shape), hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, out_dim),
            )
        else:
            raise ValueError(f"unknown mixer kind: {kind}")
        self.kind = kind

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        codes = self.net(traces).view(traces.size(0), self.n_layers, self.z_dim)
        return codes.permute(1, 0, 2).contiguous()


class WeightGenerator(nn.Module):
    def __init__(self, in_dim: int, weight_shape: tuple[int, ...], bias_size: int, hidden: int, noise_weight: float):
        super().__init__()
        self.weight_shape = tuple(int(value) for value in weight_shape)
        self.weight_numel = prod(self.weight_shape)
        self.bias_size = int(bias_size)
        self.noise_weight = float(noise_weight)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, self.weight_numel + self.bias_size),
        )

    def forward(self, code: torch.Tensor) -> Layer:
        if self.noise_weight:
            code = code + self.noise_weight * torch.randn_like(code) * 0.01
        raw = self.net(code)
        weight = raw[:, : self.weight_numel].view(-1, *self.weight_shape)
        bias = raw[:, self.weight_numel :].view(-1, self.bias_size)
        return weight, bias


def lenet_flattened_features(channels: int, image_size: int) -> int:
    sample = torch.zeros(1, channels, image_size, image_size)
    conv1 = torch.zeros(6, channels, 5, 5)
    conv2 = torch.zeros(16, 6, 5, 5)
    out = F.max_pool2d(F.relu(F.conv2d(sample, conv1, bias=torch.zeros(6))), 2)
    out = F.max_pool2d(F.relu(F.conv2d(out, conv2, bias=torch.zeros(16))), 2)
    return int(out[0].numel())


class LeNet64Target:
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int):
        channels, height, width = input_shape
        if height != width:
            raise ValueError("LeNet64 expects square inputs")
        flat = lenet_flattened_features(int(channels), int(height))
        self.weight_specs = [
            ((6, int(channels), 5, 5), 6),
            ((16, 6, 5, 5), 16),
            ((120, flat), 120),
            ((84, 120), 84),
            ((int(num_classes), 84), int(num_classes)),
        ]

    def forward(self, layers: list[Layer], inputs: torch.Tensor) -> torch.Tensor:
        out = F.max_pool2d(F.relu(F.conv2d(inputs, layers[0][0], bias=layers[0][1])), 2)
        out = F.max_pool2d(F.relu(F.conv2d(out, layers[1][0], bias=layers[1][1])), 2)
        out = out.view(out.size(0), -1)
        out = F.relu(F.linear(out, layers[2][0], bias=layers[2][1]))
        out = F.relu(F.linear(out, layers[3][0], bias=layers[3][1]))
        return F.linear(out, layers[4][0], bias=layers[4][1])


class BasicBlockFn(nn.Module):
    def __init__(self, first: Layer, second: Layer, stride: int):
        super().__init__()
        self.w1, self.b1 = first
        self.w2, self.b2 = second
        self.stride = int(stride)
        self.shortcut_pad = self.stride != 1 or self.w1.size(1) != self.w2.size(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = F.relu(F.conv2d(inputs, self.w1, bias=self.b1, stride=self.stride, padding=1))
        out = F.conv2d(out, self.w2, bias=self.b2, stride=1, padding=1)
        if self.shortcut_pad:
            shortcut = F.avg_pool2d(inputs, 2)
            pad = self.w2.size(0) // 4
            shortcut = F.pad(shortcut, (0, 0, 0, 0, pad, pad), "constant", 0)
        else:
            shortcut = inputs
        return F.relu(out + shortcut)


class ResNet32Target:
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int):
        channels, _, _ = input_shape
        self.weight_specs = [
            ((16, int(channels), 3, 3), 16),
            ((16, 16, 3, 3), 16),
            ((16, 16, 3, 3), 16),
            ((32, 16, 3, 3), 32),
            ((32, 32, 3, 3), 32),
            ((64, 32, 3, 3), 64),
            ((64, 64, 3, 3), 64),
            ((int(num_classes), 64), int(num_classes)),
        ]

    def forward(self, layers: list[Layer], inputs: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(inputs, layers[0][0], bias=layers[0][1], stride=1, padding=1)
        out = BasicBlockFn(layers[1], layers[2], stride=1)(out)
        out = BasicBlockFn(layers[3], layers[4], stride=2)(out)
        out = BasicBlockFn(layers[5], layers[6], stride=2)(out)
        out = F.avg_pool2d(out, out.size(3)).view(out.size(0), -1)
        return F.linear(out, layers[7][0], bias=layers[7][1])


def build_target(name: str, input_shape: tuple[int, int, int], num_classes: int) -> LeNet64Target | ResNet32Target:
    if name == "lenet64":
        return LeNet64Target(input_shape, num_classes)
    if name == "resnet32":
        return ResNet32Target(input_shape, num_classes)
    raise ValueError(f"unknown target: {name}")


class HyperTheft(nn.Module):
    def __init__(
        self,
        target: str,
        input_shape: tuple[int, int, int],
        trace_shape: tuple[int, ...],
        num_classes: int,
        z_dim: int = 64,
        n_seed: int = 1,
        hidden: int = 512,
        mixer: str = "auto",
        noise_weight: float = 0.0,
    ):
        super().__init__()
        self.target_name = target
        self.n_seed = int(n_seed)
        self.target = build_target(target, input_shape, num_classes)
        self.mixer = Mixer(trace_shape, z_dim, len(self.target.weight_specs), hidden, kind=mixer)
        self.generators = nn.ModuleList(
            WeightGenerator(z_dim * self.n_seed, shape, bias, hidden, noise_weight)
            for shape, bias in self.target.weight_specs
        )

    def batched_layers(self, seed_batches: list[torch.Tensor]) -> list[Layer]:
        if len(seed_batches) != self.n_seed:
            raise ValueError(f"expected {self.n_seed} seed batches, got {len(seed_batches)}")
        codes = torch.cat([self.mixer(traces) for traces in seed_batches], dim=-1)
        return [generator(codes[index]) for index, generator in enumerate(self.generators)]

    def iter_classifiers(self, seed_batches: list[torch.Tensor]) -> Iterator[list[Layer]]:
        batched = self.batched_layers(seed_batches)
        for index in range(batched[0][0].size(0)):
            yield [(weight[index], bias[index]) for weight, bias in batched]

    def classifier_logits(self, seed_batches: list[torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.target.forward(layers, inputs) for layers in self.iter_classifiers(seed_batches)])

    def forward_generated(self, seed_batches: list[torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier_logits(seed_batches, inputs).mean(dim=0)

    def config(self) -> dict:
        return {
            "target": self.target_name,
            "n_seed": self.n_seed,
            "mixer": self.mixer.kind,
            "num_layers": len(self.generators),
        }
