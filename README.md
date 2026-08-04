# Flash-VAED (ICML 2026)

Official codebase for **[Flash-VAED: Plug-and-Play VAE Decoders for Efficient Video Generation](https://github.com/Aoko955/Flash-VAED)**.

Lightweight, drop-in video VAE **decoders** for **Wan** and **LTX**. The teacher encoder is unchanged; the student decoder uses channel pruning, standard 2D convolutions at high resolution, and 3D depthwise-separable convolutions in earlier stages.

| Variant | Student checkpoint | Approx. size |
|---------|-------------------|--------------|
| Wan | `models/wan/vae_decoder_epoch21.pth` | ~23MB |
| LTX | `models/ltx/LPIPS_Best_1_14.pth` | ~944MB |

This GitHub repo hosts **inference code**. Large weights are hosted on Hugging Face (see [Download checkpoints](#download-checkpoints)).

## Repository layout

```text
Flash-VAED/
  infer.py
  requirements.txt
  models/
    wan/
      model_hybrid_aggressive.py   # Wan student
      model_original.py            # Wan teacher (encode)
    ltx/
      ltx_prune_1_4.py              # LTX student
      vendor/                       # minimal local helpers (no pip diffusers)
      teacher/config.json           # LTX teacher config (weights on HF)
```

## Install

```bash
conda create -n flashvaed python=3.10 -y
conda activate flashvaed

# Install a CUDA build of PyTorch that matches your driver, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

## Download checkpoints

Place weights under the repo root so default paths in `infer.py` resolve:

```bash
# Replace YOUR_HF_ID after you create the HF model repo, e.g. Aoko955/Flash-VAED
pip install -U "huggingface_hub[cli]"
huggingface-cli download YOUR_HF_ID \
  --local-dir . \
  --include "models/wan/*.pth" \
  --include "models/ltx/*.pth" \
  --include "models/ltx/teacher/*"
```

Expected files:

```text
models/wan/vae_decoder_epoch21.pth
models/wan/Wan2.1_VAE_orin.pth
models/ltx/LPIPS_Best_1_14.pth
models/ltx/teacher/config.json
models/ltx/teacher/diffusion_pytorch_model.safetensors
```

Override paths with `--ckpt`, `--teacher_ckpt`, or `--teacher_dir` if needed.

## Inference

### Reconstruct a video (teacher encode → student decode)

```bash
python infer.py --model wan --input demo.mp4 --output outs/wan
python infer.py --model ltx --input demo.mp4 --output outs/ltx
```

```bash
python infer.py --model wan --input demo.mp4 --output outs/wan \
  --num_frames 81 --img_h 480 --img_w 832 --fps 8 --device cuda:0
```

Outputs: `frames/*.png` and optional `recon.mp4` (disable with `--no_mp4`).

### Decode from a latent

```bash
python infer.py --model wan --mode decode --latent z.pt --output outs/decode
```

`z.pt` should be a `torch.Tensor`, or a dict with key `latent` / `latents` / `z`.

## Method sketch

**Wan student**
- Upsample stages 2–3: ~1/8 channels vs. the original Wan decoder; residuals use standard `nn.Conv2d` on `(B·T, C, H, W)`.
- Middle + upsample 0–1 residuals: 3D depthwise-separable convolutions.

**LTX student**
- Upsample stages 2–3: 1/4 channels vs. the original LTX decoder.
- Upsample 3 residuals: standard `nn.Conv2d` (spatial).
- Middle + upsample 0–1: 3D depthwise-separable convolutions.
- Upsample 2 remains pruned standard 3D.

LTX inference does **not** require `pip install diffusers`.

## Acknowledgements

- Wan VAE: [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- LTX VAE: [LTX-Video](https://huggingface.co/Lightricks/LTX-Video)

## License

Please respect upstream Wan / LTX licenses when redistributing weights. Add the project license for this release as needed.
