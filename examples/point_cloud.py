"""Example-only workflow. Run after installing topokit from any directory."""
import argparse
import json
from pathlib import Path
from demo import run_demo

parser = argparse.ArgumentParser()
parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "output"))
parser.add_argument("--plots", action="store_true")
args = parser.parse_args()
report = run_demo(args.output, plots=args.plots)
print(json.dumps(report, indent=2))
raise SystemExit(0 if all(item["status"] == "passed" for item in report["routes"].values()) else 1)
