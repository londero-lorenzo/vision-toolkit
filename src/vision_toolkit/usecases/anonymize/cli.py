from pathlib import Path

from vision_toolkit.usecases.anonymize.pipeline import (DEFAULT_FACE_MODEL, DEFAULT_POSE_MODEL, DEFAULT_SKIN_MODEL,
                                                        AnonymizeOptions, SetupPending, run)

SECONDS_PER_CHECK = 2   # a yes or a no on a confident frame
SECONDS_PER_EDIT = 60   # fixing a doubtful frame in an image editor


def add_cli(subparsers):
    p = subparsers.add_parser("anonymize", help="hide the faces in frames of a video and queue them for review",
                              description="Hide the faces in frames of a video and put the frames in a review "
                                          "queue: a person approves, edits or rejects each one before release.")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("output/anonymized"),
                   help="output root; the queue goes to OUT/<video name>/")
    p.add_argument("--step", type=float, default=300.0, help="seconds between two frames")
    p.add_argument("--times", type=parse_times, default=None,
                   help="comma-separated times in seconds, instead of --step, --start and --end")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds")
    p.add_argument("--setup", type=Path, default=None,
                   help="setup.json with crop and audience line (default OUT/<video name>/setup.json); "
                        "one file serves every video with the same layout")
    p.add_argument("--max-people", type=int, default=2, help="people the pose model looks for (its num_poses)")
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL,
                   help="pose model from the catalog (see `vision-toolkit models list`)")
    p.add_argument("--face-model", type=none_or_name, default=DEFAULT_FACE_MODEL,
                   help="face detection model from the catalog, or none")
    p.add_argument("--skin-model", type=none_or_name, default=DEFAULT_SKIN_MODEL,
                   help="segmentation model with a face-skin label, run around each head, or none")
    p.add_argument("--blocks", type=int, default=6, help="pixelation blocks across a face")
    p.add_argument("--review-budget", type=int, default=None,
                   help="most doubtful episodes to queue, evenly spread in time (default: all)")
    p.add_argument("--substitute-window", type=float, default=3.0,
                   help="seconds around a doubtful frame searched for a confident one")
    p.add_argument("--episode-gap", type=float, default=None,
                   help="doubtful frames at most this many seconds apart form one episode "
                        "(default: twice the step, 0 with --times)")
    p.add_argument("--min-blob", type=float, default=0.002,
                   help="smallest unexplained foreground, as a fraction of the frame, that makes a frame doubtful")
    p.set_defaults(func=main)


def parse_times(value):
    return tuple(float(v) for v in value.split(","))


def none_or_name(value):
    return None if value.lower() == "none" else value


def main(args):
    options = AnonymizeOptions(
        step=args.step, times=args.times, start=args.start, end=args.end, setup=args.setup,
        max_people=args.max_people, pose_model=args.pose_model, face_model=args.face_model,
        skin_model=args.skin_model, blocks=args.blocks, review_budget=args.review_budget,
        substitute_window=args.substitute_window, episode_gap=args.episode_gap, min_blob=args.min_blob,
    )
    try:
        summary = run(args.video, args.out, options)
    except SetupPending as e:
        raise SystemExit(str(e))
    print(f"Queue in {summary.out}")
    print(f"auto: {summary.auto} frames queued, about {minutes(summary.auto * SECONDS_PER_CHECK)} to check")
    print(f"doubtful: {summary.doubtful} frames in {summary.episodes} episodes, {summary.doubtful_queued} queued "
          f"(about {minutes(summary.doubtful_queued * SECONDS_PER_EDIT)} to edit), {summary.skipped} skipped")
    if summary.decided:
        print(f"already decided: {summary.decided}")


def minutes(seconds):
    return f"{max(1, round(seconds / 60))} min" if seconds else "0 min"
