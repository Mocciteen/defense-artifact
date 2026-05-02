from __future__ import annotations

import argparse
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module, latent_dim: int):
        super().__init__()
        self.module = module
        self.latent_dim = int(latent_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        output = self.module(latents)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, dict):
            output = output.get("image", output.get("images"))
        if not torch.is_tensor(output):
            raise TypeError("GAN prior must return tensor-like image output")
        return output


def import_symbol(module_path: str, symbol_name: str):
    spec = importlib.util.spec_from_file_location("artifact_external_gan", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import GAN module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


def load_gan_prior(args: argparse.Namespace, device: torch.device) -> GANModuleWrapper | None:
    if not args.gan_prior:
        return None
    if args.gan_module and args.gan_class:
        gan_cls = import_symbol(args.gan_module, args.gan_class)
        module = gan_cls()
        checkpoint = torch.load(args.gan_prior, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        module.load_state_dict(state_dict, strict=False)
    else:
        module = torch.load(args.gan_prior, map_location="cpu", weights_only=False)
        if isinstance(module, dict) and isinstance(module.get("module"), nn.Module):
            module = module["module"]
        if not isinstance(module, nn.Module):
            raise TypeError("GAN checkpoint must contain an nn.Module, or use --gan-module and --gan-class")
    module.eval().to(device)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return GANModuleWrapper(module, args.gan_latent_dim).to(device)


def optimize_gan_latents(
    gan: GANModuleWrapper,
    targets: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        generator = torch.Generator(device=targets.device)
        generator.manual_seed(int(seed))
        latents = torch.randn(targets.shape[0], gan.latent_dim, generator=generator, device=targets.device)
        latents.requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=lr)
        for _ in range(max(0, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            prior_images = gan(latents)
            loss = F.mse_loss(prior_images, targets)
            loss.backward()
            optimizer.step()
        prior_images = gan(latents).detach()
    prior_loss = F.mse_loss(prior_images, targets.detach(), reduction="none").flatten(1).mean(dim=1)
    return prior_images, prior_loss


def apply_gan_projection(
    images: torch.Tensor,
    gan: GANModuleWrapper | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if gan is None or int(args.gan_project_steps) <= 0:
        return images, None
    prior_images, prior_loss = optimize_gan_latents(
        gan,
        images.detach(),
        steps=args.gan_project_steps,
        lr=args.gan_project_lr,
        seed=args.seed,
    )
    blend = float(args.gan_blend)
    projected = torch.clamp((1.0 - blend) * images + blend * prior_images, min=-1.0, max=1.0)
    return projected, prior_loss
