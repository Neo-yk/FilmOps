#!/usr/bin/env python3
"""Character Layout — multi-image mode: keyframe + character reference crops.

Example:
    python examples/character_layout_multi_image.py \\
        --frame frame.jpg --characters char0.jpg char1.jpg \\
        --ckpt-dir ./checkpoints
"""

import sys
import time

from _common import base_parser, build_pipeline, print_and_save


def main() -> int:
    parser = base_parser("FilmOps — character layout (multi-image)")
    parser.add_argument("--frame", required=True, help="Keyframe image path.")
    parser.add_argument("--characters", nargs="+", default=[],
                        help="Character reference crop image paths (zero or more).")
    args = parser.parse_args()

    # Force the right operator.
    if not args.operators:
        args.operators = "character_layout"

    pipe = build_pipeline(args)
    op = pipe.operators.get("character_layout")
    if op is None:
        print("Error: character_layout operator is not loaded.")
        return 1

    payload = {"frame": args.frame, "characters": args.characters} \
        if args.characters else args.frame

    print(f"Analysing keyframe: {args.frame} (+{len(args.characters)} character refs)\n")
    t0 = time.time()
    result = {"character_layout": op.predict(payload)}
    print(f"  Inference time: {time.time() - t0:.2f}s")

    print_and_save(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
