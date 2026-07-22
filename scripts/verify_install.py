#!/usr/bin/env python3
"""Print every registered operator and check its weight paths.

Usage:
    python scripts/verify_install.py --ckpt-dir ./checkpoints
"""

import argparse
import os
import sys

from filmops import FilmOpsConfig, list_operators


def main() -> int:
    p = argparse.ArgumentParser(description="Verify FilmOps install / checkpoints")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    args = p.parse_args()

    cfg = FilmOpsConfig(checkpoint_dir=args.ckpt_dir)
    print(f"Checkpoint root: {os.path.abspath(args.ckpt_dir)}\n")
    print(f"{'Operator':<22}  {'Required path':<60}  Status")
    print("-" * 100)

    missing = 0
    for name in list_operators():
        try:
            kwargs = cfg.operator_configs.load_kwargs(
                name, base_checkpoint_dir=args.ckpt_dir, device="cpu",
            )
        except KeyError:
            print(f"{name:<22}  <no load_kwargs builder>")
            continue

        for key, val in kwargs.items():
            if not isinstance(val, str) or key == "device":
                continue
            exists = os.path.exists(val)
            mark = "OK " if exists else "MISS"
            if not exists:
                missing += 1
            print(f"{name:<22}  {val:<60}  [{mark}]")
        print()

    if missing:
        print(f"⚠ {missing} path(s) missing.")
        return 1
    print("✓ All required paths present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
