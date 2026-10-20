import argparse
import json
from pathlib import Path

from vision_toolkit.usecases.board.person_mask import DEFAULT_POSE_MODEL
from vision_toolkit.usecases.board.pipeline import BoardNotFoundError, BoardOptions, SetupError, run


def add_cli(subparsers):
    p = subparsers.add_parser("board", help="remove the lecturer from the board in a lecture video",
                              description="Remove the lecturer from the board in a lecture video: clean snapshots "
                                          "at every change and a log of the pointed spot.")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("output"), help="output root; results go to OUT/<video name>/")
    p.add_argument("--step", type=float, default=2.0, help="seconds between two analysed frames")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds")
    p.add_argument("--setup", type=Path, default=None,
                   help="confirmed setup.json (from `vision-toolkit anonymize`) with crop and audience line")
    p.add_argument("--crop", type=parse_crop, default=None,
                   help="x,y,w,h of the video area in a screen recording (overrides the setup)")
    p.add_argument("--redact-below", type=int, default=None,
                   help="audience line in the cropped frame: rows from here down are blacked out before the pose "
                        "model sees the frame (overrides the setup)")
    p.add_argument("--quad", type=Path, default=None, help="JSON with the 4 board corners (skips detection)")
    p.add_argument("--window", type=int, default=21, help="samples in the temporal median")
    p.add_argument("--min-seen", type=float, default=0.25,
                   help="fraction of the window a board pixel must be seen in to change; below it, the pixel "
                        "keeps its last value")
    p.add_argument("--min-change", type=float, default=0.004,
                   help="fraction of changed pixels that triggers a new snapshot")
    p.add_argument("--change-threshold", type=float, default=None,
                   help="levels a pixel must change by to count as changed (default: half the chalk contrast "
                        "measured on the video, between 6 and 25)")
    p.add_argument("--match-exposure", action=argparse.BooleanOptionalAction, default=True,
                   help="divide each frame by its exposure gain against the median of its framing segment")
    p.add_argument("--framing-step", type=none_or_float, default=15.0,
                   help="seconds between two frames of the camera framing check, or none to skip it")
    p.add_argument("--min-segment", type=float, default=60.0,
                   help="framing segments shorter than this many seconds are skipped")
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL,
                   help="pose model from the catalog (see `vision-toolkit models list`)")
    p.add_argument("--pose-confidence", type=none_or_float, default=0.1,
                   help="min_confidence passed to the pose model, or none for the model's default")
    p.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True,
                   help="when the pose finds nobody, mask the foreground blob shaped like a standing person")
    p.add_argument("--grow-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="add to the pose mask the foreground blobs that touch it (a sleeve, a shadow)")
    p.add_argument("--track", type=float, default=20.0,
                   help="seconds the lecturer is followed through the foreground touching her last mask, when "
                        "neither the pose nor the standing shape finds her (0: never)")
    p.add_argument("--refine-model", default=None,
                   help="segmentation model run on the crop around the lecturer and merged into the mask, "
                        "e.g. deeplab-v3")
    p.add_argument("--refine-labels", type=lambda v: tuple(v.split(",")), default=("person",),
                   help="comma-separated labels of the refine model that belong to the lecturer (default: person)")
    p.set_defaults(func=main)


def parse_crop(value):
    return tuple(int(v) for v in value.split(","))


def none_or_float(value):
    return None if value.lower() == "none" else float(value)


def main(args):
    options = BoardOptions(
        step=args.step, start=args.start, end=args.end, crop=args.crop, redact_below=args.redact_below,
        setup=args.setup, quad=args.quad, window=args.window, min_seen=args.min_seen, min_change=args.min_change,
        change_threshold=args.change_threshold, match_exposure=args.match_exposure,
        framing_step=args.framing_step, min_segment=args.min_segment, pose_model=args.pose_model,
        pose_confidence=args.pose_confidence, fallback=args.fallback, grow_mask=args.grow_mask, track=args.track,
        refine_model=args.refine_model,
        refine_labels=args.refine_labels,
    )
    try:
        out = run(args.video, args.out, options)
    except (BoardNotFoundError, SetupError) as e:
        raise SystemExit(str(e))
    print(f"Output in {out}")
    summary = json.loads((out / "run.json").read_text(encoding="utf-8"))
    used = [f for f in summary["framings"] if f["used"]]
    skipped = [f for f in summary["framings"] if not f["used"]]
    print(f"framing: {len(used)} segment(s) used, {len(skipped)} skipped")
    print(f"snapshots: {summary['snapshots']}, change threshold {summary['change_threshold']}")
    frames = summary["frames"]
    if frames["analysed"]:
        share = {k: round(100 * v / frames["analysed"]) for k, v in frames.items() if k != "analysed"}
        print(f"lecturer found by pose in {share['lecturer_by_pose']}% of the frames, "
              f"by foreground in {share['lecturer_by_foreground']}%, by tracking in {share['lecturer_by_tracking']}%, "
              f"nobody in {share['nobody']}%")
