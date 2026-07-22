#!/usr/bin/env python3
"""Analyse a single image with all enabled frame-level operators.

Example:
    python examples/analyse_image.py --image frame.jpg --ckpt-dir ./checkpoints
"""

import sys
import time

from _common import base_parser, build_pipeline, print_and_save


def main() -> int:
    parser = base_parser("FilmOps — single image analysis")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    args = parser.parse_args()

    pipe = build_pipeline(args)
    print(f"Analysing image: {args.image}\n")
    t0 = time.time()
    result = pipe.analyse_image(args.image)
    print(f"  Inference time: {time.time() - t0:.2f}s")

    print_and_save(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
