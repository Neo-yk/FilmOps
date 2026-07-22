# `checkpoints/` — Model weights

> ⚠ **This directory holds binary weight files only. Do not commit Python source code here.**

## Required layout

```
checkpoints/
├── shot_scale/
│   ├── dinov2_shot_scale.pth
│   └── dinov2_vitb14_pretrain.pth
├── composition/
│   ├── dinov3_composition.pth
│   └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── camera_angle/
│   ├── shot_angle/
│   └── dutch_shot/
├── color_tone/
│   └── resnet18_color_tone.pth
├── character_layout/
│   └── character_layout_ckpt/
│       ├── config.json
│       ├── modeling_*.py
│       └── ...
└── camera_movement/
    └── camera_movement_ckpt/
        ├── config.json
        ├── modeling_*.py
        └── ...
```

## Download

### Step 1: Download official pretrained backbones 

These are **not** included in our HF repo — download them directly from Meta:

```bash
# DINOv2 ViT-B/14 (for Shot Scale) — public link
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
  -O checkpoints/shot_scale/dinov2_vitb14_pretrain.pth

# DINOv3 ViT-B/16 (for Composition) — apply at the official page, then download the
# "ViT-B/16 distilled / LVD-1689M" weight via the URL in the confirmation e-mail,
# and rename to `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`:
#   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
cp <YOUR_DOWNLOADED_FILE> checkpoints/composition/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### Step 2: Download fine-tuned weights from HuggingFace

```bash
huggingface-cli download your-org/filmops-checkpoints --local-dir ./checkpoints
```

Or download individual files manually from the HF model page.

## Weight licenses

| File | Source | License |
|------|--------|--------|
| `dinov2_vitb14_pretrain.pth` | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 — **download from official** |
| `dinov2_shot_scale.pth` | Fine-tuned from DINOv2 | Apache-2.0 |
| `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | [DINOv3 License](../filmops/backbones/dinov3/LICENSE.md) — **download from official** |
| `dinov3_composition.pth` | Fine-tuned from DINOv3 | [DINOv3 License](../filmops/backbones/dinov3/LICENSE.md) |
| `camera_angle/*/pytorch_model.bin` | Fine-tuned from BEiT ([microsoft/unilm](https://github.com/microsoft/unilm)) | MIT |
| `resnet18_color_tone.pth` | Trained from scratch | Apache-2.0 |
| `character_layout/character_layout_ckpt/*` | Fine-tuned from InternVL3 ([OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)) | MIT |
| `camera_movement/camera_movement_ckpt/*` | Fine-tuned from InternVL3 | MIT |

> ⚠ Weights derived from the DINOv3 backbone are subject to the DINOv3 License
> Agreement. By downloading or using these weights, you agree to its terms.
> See [`filmops/backbones/dinov3/LICENSE.md`](../filmops/backbones/dinov3/LICENSE.md).

## Verifying

```bash
python scripts/verify_install.py --ckpt-dir ./checkpoints
```
prints every operator's required paths and marks each as `OK` or `MISS`.
