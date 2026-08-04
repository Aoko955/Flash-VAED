from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent
WAN_DIR = ROOT / "models" / "wan"
LTX_DIR = ROOT / "models" / "ltx"

LOCAL_DIFFUSERS_SRC = LTX_DIR / "vendor"

DEFAULTS = {
    "wan": {
        "student_ckpt": WAN_DIR / "vae_decoder_epoch21.pth",
        "teacher_ckpt": WAN_DIR / "Wan2.1_VAE_orin.pth",
    },
    "ltx": {
        "student_ckpt": LTX_DIR / "LPIPS_Best_1_14.pth",
        "teacher_dir": LTX_DIR / "teacher",
    },
}

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("flash_vaed")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_local_diffusers():

    src = str(LOCAL_DIFFUSERS_SRC)
    if not (LOCAL_DIFFUSERS_SRC / "diffusers").is_dir():
        raise FileNotFoundError(
            f"Local LTX vendor not found at {LOCAL_DIFFUSERS_SRC / 'diffusers'}"
        )
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)

    stale = [k for k in list(sys.modules) if k == "diffusers" or k.startswith("diffusers.")]
    for k in stale:
        mod = sys.modules.get(k)
        f = getattr(mod, "__file__", None) or ""
        if "Flash-VAED/models/ltx/vendor" not in f.replace("\\", "/"):
            del sys.modules[k]


def _load_ltx_teacher(teacher_dir: Path, device: torch.device):

    import json

    from safetensors.torch import load_file
    from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo

    cfg_path = teacher_dir / "config.json"
    weight_path = teacher_dir / "diffusion_pytorch_model.safetensors"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    skip = {"_class_name", "_diffusers_version"}
    ctor = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k not in skip}
    model = AutoencoderKLLTXVideo(**ctor)
    state = load_file(str(weight_path))
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def load_video(
    path: str,
    num_frames: int,
    img_size: Tuple[int, int],
) -> torch.Tensor:

    try:
        from decord import VideoReader
    except ImportError as e:
        raise ImportError("Please install decord: pip install decord") from e

    vr = VideoReader(path)
    total = len(vr)
    if total < num_frames:
        indices = np.pad(np.arange(total), (0, num_frames - total), mode="wrap")
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)

    video = torch.from_numpy(vr.get_batch(indices).asnumpy())
    if video.shape[-1] > 3:
        video = video[..., :3]
    elif video.shape[-1] == 1:
        video = torch.cat([video] * 3, dim=-1)

    video = video.permute(0, 3, 1, 2).float() / 255.0
    video = T.Compose([T.Resize(img_size), T.CenterCrop(img_size)])(video)
    video = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(video)
    return video.permute(1, 0, 2, 3)


def save_video_tensor(
    video: torch.Tensor,
    out_dir: Path,
    save_frames: bool = True,
    save_mp4: bool = True,
    fps: int = 8,
):

    out_dir.mkdir(parents=True, exist_ok=True)
    x = video.detach().float().cpu().clamp(-1, 1)
    x01 = (x + 1.0) / 2.0

    if save_frames:
        frame_dir = out_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for t in range(x01.shape[1]):
            torchvision.utils.save_image(x01[:, t], frame_dir / f"{t:04d}.png")

    if save_mp4:
        frames = (x01.permute(1, 2, 3, 0).numpy() * 255.0).round().astype(np.uint8)
        mp4_path = out_dir / "recon.mp4"
        try:
            torchvision.io.write_video(str(mp4_path), torch.from_numpy(frames), fps=fps)
            log.info(f"Saved video: {mp4_path}")
        except Exception as e:
            log.warning(f"Failed to write mp4 ({e}); frames were still saved if enabled.")

    log.info(f"Saved outputs under: {out_dir}")


class WanFlashVAED:
    def __init__(
        self,
        student_ckpt: Path,
        teacher_ckpt: Optional[Path],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        wan_student = _load_module("flash_vaed_wan_student", WAN_DIR / "model_hybrid_aggressive.py")
        self.student = wan_student.WanVAE(vae_pth=str(student_ckpt), device=device, dtype=dtype)
        self.teacher = None
        self.device = device
        if teacher_ckpt is not None:
            wan_teacher = _load_module("flash_vaed_wan_teacher", WAN_DIR / "model_original.py")
            self.teacher = wan_teacher.WanVAE(
                vae_pth=str(teacher_ckpt), device=str(device), dtype=dtype
            )

    @torch.no_grad()
    def encode(self, video_cthw: torch.Tensor) -> torch.Tensor:
        if self.teacher is None:
            raise RuntimeError("Wan teacher not loaded; pass --teacher_ckpt for reconstruct mode")

        latents = self.teacher.encode([video_cthw.to(self.device)])
        z = latents[0] if isinstance(latents, list) else latents
        return z

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:

        out = self.student.decode(latent.to(self.device))
        if isinstance(out, list):
            out = out[0]
        if out.ndim == 5:
            out = out[0]
        return out.clamp_(-1, 1)


class LTXFlashVAED:
    def __init__(
        self,
        student_ckpt: Path,
        teacher_dir: Optional[Path],
        device: torch.device,
    ):
        _ensure_local_diffusers()

        ltx_mod = _load_module("flash_vaed_ltx_student", LTX_DIR / "ltx_prune_1_4.py")
        self.student = ltx_mod.AutoencoderKLLTXVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=128,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=(4, 3, 3, 3, 4),
            patch_size=4,
            patch_size_t=1,
        ).to(device)
        sd = torch.load(str(student_ckpt), map_location="cpu")
        for k in ("mean_of_means", "std_of_means"):
            sd.pop(k, None)
        self.student.load_state_dict(sd, strict=False)
        self.student.eval()
        self.device = device

        self.teacher = None
        if teacher_dir is not None:
            import diffusers as _df

            log.info(f"Using local LTX helpers from {_df.__file__}")
            self.teacher = _load_ltx_teacher(Path(teacher_dir), device)
            if hasattr(self.teacher, "latents_mean") and hasattr(self.teacher, "latents_std"):
                self.student.register_buffer(
                    "latents_mean", self.teacher.latents_mean.clone()
                )
                self.student.register_buffer(
                    "latents_std", self.teacher.latents_std.clone()
                )

    @torch.no_grad()
    def encode(self, video_cthw: torch.Tensor) -> torch.Tensor:
        if self.teacher is None:
            raise RuntimeError("LTX teacher not loaded; pass --teacher_dir for reconstruct mode")
        videos = video_cthw.unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=self.device.type == "cuda"):
            latents = self.teacher.encode(videos).latent_dist.sample()
        return latents

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        latent = latent.to(self.device)
        scale = getattr(self.student.config, "scaling_factor", 1.0)
        latent = latent / scale
        temb = torch.zeros(latent.shape[0], device=self.device, dtype=latent.dtype)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=self.device.type == "cuda"):
            out = self.student.decode(latent, temb=temb, return_dict=False)[0]
        return out[0].clamp_(-1, 1)


def build_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.model == "wan":
        teacher = Path(args.teacher_ckpt) if args.teacher_ckpt else None
        if args.mode == "reconstruct" and teacher is None:
            teacher = DEFAULTS["wan"]["teacher_ckpt"]
        return WanFlashVAED(
            student_ckpt=Path(args.ckpt),
            teacher_ckpt=teacher if args.mode == "reconstruct" else None,
            device=device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ), device
    if args.model == "ltx":
        teacher = Path(args.teacher_dir) if args.teacher_dir else None
        if args.mode == "reconstruct" and teacher is None:
            teacher = DEFAULTS["ltx"]["teacher_dir"]
        return LTXFlashVAED(
            student_ckpt=Path(args.ckpt),
            teacher_dir=teacher if args.mode == "reconstruct" else None,
            device=device,
        ), device
    raise ValueError(f"Unknown model: {args.model}")


def parse_args():
    p = argparse.ArgumentParser(description="Flash-VAED inference (Wan / LTX)")
    p.add_argument("--model", choices=["wan", "ltx"], required=True)
    p.add_argument("--mode", choices=["reconstruct", "decode"], default="reconstruct")
    p.add_argument("--input", type=str, default=None, help="Input video for reconstruct")
    p.add_argument("--latent", type=str, default=None, help="Latent .pt/.pth for decode mode")
    p.add_argument("--output", type=str, required=True, help="Output directory")
    p.add_argument("--ckpt", type=str, default=None, help="Student checkpoint override")
    p.add_argument("--teacher_ckpt", type=str, default=None, help="Wan teacher ckpt")
    p.add_argument("--teacher_dir", type=str, default=None, help="LTX teacher folder")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=832)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no_mp4", action="store_true")
    p.add_argument("--no_frames", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ckpt is None:
        args.ckpt = str(
            DEFAULTS["wan"]["student_ckpt"]
            if args.model == "wan"
            else DEFAULTS["ltx"]["student_ckpt"]
        )

    if args.mode == "reconstruct" and not args.input:
        raise SystemExit("--input video is required for reconstruct mode")
    if args.mode == "decode" and not args.latent:
        raise SystemExit("--latent is required for decode mode")

    log.info(f"Model={args.model} mode={args.mode} ckpt={args.ckpt}")
    engine, device = build_model(args)

    if args.mode == "reconstruct":
        video = load_video(args.input, args.num_frames, (args.img_h, args.img_w))
        log.info(f"Loaded video {args.input} -> {tuple(video.shape)}")
        latent = engine.encode(video)
        if isinstance(latent, torch.Tensor):
            log.info(f"Latent shape: {tuple(latent.shape)}")
        recon = engine.decode(latent)
    else:
        latent = torch.load(args.latent, map_location="cpu")
        if isinstance(latent, dict):

            for key in ("latent", "latents", "z"):
                if key in latent:
                    latent = latent[key]
                    break
        if not isinstance(latent, torch.Tensor):
            raise TypeError(f"Unsupported latent type: {type(latent)}")
        log.info(f"Loaded latent -> {tuple(latent.shape)}")
        recon = engine.decode(latent)

    log.info(f"Recon shape: {tuple(recon.shape)}")
    save_video_tensor(
        recon,
        Path(args.output),
        save_frames=not args.no_frames,
        save_mp4=not args.no_mp4,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
