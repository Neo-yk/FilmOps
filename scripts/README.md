# Scripts

Operational / debugging scripts that are **not** part of the installable
Python package. Run them directly with `python scripts/<name>.py`.

| Script | Purpose |
|--------|---------|
| `verify_install.py` | Print every registered operator and check that its weight directory exists. |
| `download_checkpoints.py` | Download pretrained backbones from official sources and finetuned weights from HuggingFace. |
