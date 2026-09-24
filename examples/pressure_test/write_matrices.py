"""Export the declared experiment plan without running scientific workloads."""
import csv
import json
from pathlib import Path

from cases import build_cases, build_compact_cases


def main():
    directory = Path(__file__).with_name("matrices")
    directory.mkdir(exist_ok=True)
    matrices = {suite: build_cases(suite=suite) for suite in ("smoke", "pressure", "supplemental")}
    matrices.update({f"pressure_q{degree}": [case for case in matrices["pressure"]
                     if case["degree"] == degree] for degree in range(3)})
    matrices.update(compact_smoke=build_compact_cases(smoke=True), compact_pressure=build_compact_cases())
    for name, cases in matrices.items():
        (directory / f"{name}.json").write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
        with (directory / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(cases[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                              for key, value in case.items()} for case in cases)
        print(f"{name}: {len(cases)} cases")


if __name__ == "__main__":
    main()
