# FilmOps Examples

End-to-end examples that show how to invoke the FilmOps pipeline.

| Script | Purpose |
|--------|---------|
| `analyse_image.py` | Run all frame-level operators on a single image. |
| `analyse_video.py` | Run shot-level operators (camera_movement) on a video. |
| `character_layout_multi_image.py` | Use Character Layout in multi-image mode (frame + character reference crops). |

All scripts share the same CLI flags:

```
--ckpt-dir   Root directory of model weights (default: ./checkpoints)
--device     cuda | cpu                       (default: cuda)
--operators  comma-separated subset of operators to enable
--output     Optional JSON output path
```
