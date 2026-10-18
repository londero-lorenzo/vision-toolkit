from pathlib import Path

from vision_toolkit.usecases.board.person_mask import DEFAULT_POSE_MODEL
from vision_toolkit.usecases.board.pipeline import BoardNotFoundError, BoardOptions, run


def add_cli(subparsers):
    p = subparsers.add_parser("board", help="remove the lecturer from the board in a lecture video",
                              description="Remove the lecturer from the board in a lecture video: clean snapshots "
                                          "at every change and a log of the pointed spot.")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("output"), help="output root; results go to OUT/<video name>/")
    p.add_argument("--step", type=float, default=1.0, help="seconds between two analysed frames")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds")
    p.add_argument("--crop", type=parse_crop, default=None, help="x,y,w,h of the video area in a screen recording")
    p.add_argument("--quad", type=Path, default=None, help="JSON with the 4 board corners (skips detection)")
    p.add_argument("--window", type=int, default=7, help="samples in the temporal median")
    p.add_argument("--min-change", type=float, default=0.004,
                   help="fraction of changed pixels that triggers a new snapshot")
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL,
                   help="pose model from the catalog (see `vision-toolkit models list`)")
    p.add_argument("--refine-model", default=None,
                   help="segmentation model run on the crop around the lecturer and merged into the mask, "
                        "e.g. deeplab-v3")
    p.add_argument("--refine-labels", type=lambda v: tuple(v.split(",")), default=("person",),
                   help="comma-separated labels of the refine model that belong to the lecturer (default: person)")
    p.set_defaults(func=main)


def parse_crop(value):
    return tuple(int(v) for v in value.split(","))


def main(args):
    options = BoardOptions(
        step=args.step, start=args.start, end=args.end, crop=args.crop, quad=args.quad,
        window=args.window, min_change=args.min_change, pose_model=args.pose_model,
        refine_model=args.refine_model, refine_labels=args.refine_labels,
    )
    try:
        out = run(args.video, args.out, options)
    except BoardNotFoundError as e:
        raise SystemExit(str(e))
    print(f"Output in {out}")
