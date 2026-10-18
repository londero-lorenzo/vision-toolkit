import argparse

from vision_toolkit import __version__
from vision_toolkit.models import catalog
from vision_toolkit.usecases.board import cli as board_cli

# Each use case adds its own subcommand
USECASES = (board_cli,)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vision-toolkit", description="Image and video processing toolkit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    add_models_cli(commands)
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


if __name__ == "__main__":
    main()
