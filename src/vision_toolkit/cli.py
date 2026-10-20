import argparse
from pathlib import Path

from vision_toolkit import __version__
from vision_toolkit.core.review import ReviewQueue
from vision_toolkit.models import catalog
from vision_toolkit.usecases.anonymize import cli as anonymize_cli
from vision_toolkit.usecases.board import cli as board_cli

# Each use case adds its own subcommand
USECASES = (board_cli, anonymize_cli)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vision-toolkit", description="Image and video processing toolkit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    add_models_cli(commands)
    add_review_cli(commands)
    for usecase in USECASES:
        usecase.add_cli(commands)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except catalog.CatalogError as e:
        raise SystemExit(str(e))


def add_models_cli(commands):
    p = commands.add_parser("models", help="list and download the models in the catalog")
    actions = p.add_subparsers(dest="action", required=True)

    ls = actions.add_parser("list", help="list the models, their task and whether their files are downloaded")
    ls.add_argument("--task", default=None, help="only models of this task (segmentation, pose, ...)")
    ls.set_defaults(func=models_list)

    dl = actions.add_parser("download", help="download model files into the models folder")
    dl.add_argument("names", nargs="*", help="models to download (default: all)")
    dl.set_defaults(func=models_download)


def models_list(args):
    for spec in catalog.available(args.task):
        status = "not downloaded" if catalog.missing_files(spec) else "ready"
        print(f"{spec.name:22s} {spec.task:13s} {spec.backend:10s} {status}")
        if spec.description:
            print(f"    {spec.description}")


def models_download(args):
    catalog.download(args.names)


def add_review_cli(commands):
    p = commands.add_parser("review", help="review queues: images waiting for a person's decision")
    actions = p.add_subparsers(dest="action", required=True)

    add = actions.add_parser("add", help="put images in QUEUE/todo/<group>/, named after their files")
    add.add_argument("queue", type=Path)
    add.add_argument("images", nargs="+", type=Path)
    add.add_argument("--group", required=True, help="group of the items, e.g. snapshots")
    add.add_argument("--question", required=True, help="what the person decides, e.g. 'Is the lecturer absent?'")
    add.set_defaults(func=review_add)

    st = actions.add_parser("status", help="count the items of a queue per group and state")
    st.add_argument("queue", type=Path)
    st.set_defaults(func=review_status)


def review_add(args):
    queue = ReviewQueue(args.queue)
    for image in args.images:
        added = queue.add(image, image.stem, args.group, args.question)
        print(f"{'added' if added else 'kept, already decided or edited'}: {image.stem}")


def review_status(args):
    counts = ReviewQueue(args.queue).status()
    if not counts:
        print(f"No items in {args.queue}")
        return
    states = ("todo", "approved", "edited", "rejected", "missing")
    print(f"{'group':14s}" + "".join(f"{s:>10s}" for s in states))
    for group, c in sorted(counts.items()):
        print(f"{group:14s}" + "".join(f"{c[s]:10d}" for s in states))


if __name__ == "__main__":
    main()
