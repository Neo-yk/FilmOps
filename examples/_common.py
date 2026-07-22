"""Shared CLI helpers for the example scripts."""

import argparse
import time

from filmops import FilmOpsConfig, FilmOpsPipeline
from filmops.utils.logging import configure_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-dir", default="./checkpoints",
                   help="Root directory for model checkpoints.")
    p.add_argument("--device", default="cuda", help="Inference device.")
    p.add_argument("--operators", default=None,
                   help="Comma-separated list of operators to enable. "
                        "Default: all registered.")
    p.add_argument("--output", default=None,
                   help="Optional JSON output path.")
    return p


def build_pipeline(args) -> FilmOpsPipeline:
    configure_logging()
    enabled = args.operators.split(",") if args.operators else None
    cfg = FilmOpsConfig(
        checkpoint_dir=args.ckpt_dir,
        device=args.device,
        enabled_operators=enabled,
    )
    pipe = FilmOpsPipeline(cfg)
    print("=" * 60)
    print("  FilmOps — Operator Toolkit for FilmBench")
    print("=" * 60)
    print(f"  Checkpoint dir : {args.ckpt_dir}")
    print(f"  Device         : {args.device}")
    print(f"  Operators      : {enabled or 'all'}")
    print("=" * 60)

    t0 = time.time()
    pipe.load()
    print(f"\nModels loaded in {time.time() - t0:.2f}s\n")
    return pipe


def print_and_save(result: dict, output_path: str | None) -> None:
    print("-" * 60)
    print("  RESULTS")
    print("-" * 60)
    for name, res in result.items():
        labels = res.get("labels", res.get("error", "N/A"))
        if isinstance(labels, list):
            labels_str = ", ".join(str(x) for x in labels)
        else:
            labels_str = str(labels)
        print(f"  [{name:20s}] {labels_str}")
    print("-" * 60)

    if output_path:
        from filmops.utils.io import dump_json
        dump_json(result, output_path)
        print(f"\nResults saved to: {output_path}")
