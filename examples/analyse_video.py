#!/usr/bin/env python3
"""Analyse a video with shot-level operators (camera_movement).

Example:
    python examples/analyse_video.py --video shot.mp4 --ckpt-dir ./checkpoints
"""

import sys
import time

from _common import base_parser, build_pipeline, print_and_save


def main() -> int:
    parser = base_parser("FilmOps — single video analysis")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    args = parser.parse_args()

    pipe = build_pipeline(args)
    print(f"Analysing video: {args.video}\n")
    t0 = time.time()
    result = pipe.analyse_video(args.video)
    print(f"  Inference time: {time.time() - t0:.2f}s")

    print_and_save(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
