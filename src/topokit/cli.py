"""Local CLI; never installs packages, contacts a service, or publishes data."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="topokit", description="Native point-cloud and explicit-object topology")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("info", help="show installed version and release scope")
    args = parser.parse_args(argv)
    if args.command == "info":
        from . import __version__
        print(json.dumps({"name": "topokit", "version": __version__, "cores": ["simplicial", "hyperdigraph", "interaction"],
                          "layers": ["readers", "builders", "core", "postprocessing", "visualization", "workflows"],
                          "interaction_factors": 2, "weighted_alpha": False, "path_homology": False}, indent=2))
        return 0
