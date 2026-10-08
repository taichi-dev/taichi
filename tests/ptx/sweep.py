"""Screen compiler experiments independently against unchanged PTX gold.

Uses the current PYTHONPATH and solver Python environment. Runs serially to avoid
timing interference. A failed candidate is recorded and subsequent candidates
still run; all output and native-library provenance are retained.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simfinity-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=[
        "control", "cse-repair", "cse-local", "ast-unused",
        "lazy-forwarding", "shared-forwarding", "control",
    ], help="Use '+' to combine experiments in one variant")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, variant in enumerate(args.variants):
        directory = output / f"{index:02d}-{variant}"
        command = [
            sys.executable, "-m", "pytest", str(Path(__file__).resolve().parent), "-v",
            "--simfinity-repo", str(args.simfinity_repo.resolve()),
            "--ptx-mode", "optimized", "--ptx-profile", "--basetemp", str(directory),
        ]
        if variant != "control":
            for experiment in variant.split("+"):
                command += ["--ptx-experiment", experiment]
        print(f"Starting {variant}", flush=True)
        with directory.with_suffix(".log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = {"variant": variant, "passed": process.returncode == 0, "operators": {}}
        for path in sorted({path.resolve() for path in directory.glob("test_*/result.json")}):
            result = json.loads(path.read_text())
            log = re.sub(r"\x1b\[[0-9;]*m", "", (path.parent / "compile.log").read_text())
            passes = {}
            for value, unit, name in re.findall(r"([\d.]+)\s+(us|ms|s)\s+[\d.]+%\s+(\w+)", log):
                passes[name] = passes.get(name, 0.0) + float(value) * {"us": 1e-6, "ms": 1e-3, "s": 1}[unit]
            result["profile_seconds"] = passes
            result["native_seconds"] = sum(r["seconds"] for r in result["records"])
            row["operators"][result["method"]] = result
        row["native_seconds"] = sum(r["native_seconds"] for r in row["operators"].values())
        results.append(row)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{variant}: passed={row['passed']}, native={row['native_seconds']:.3f} s", flush=True)
    if not all(row["passed"] for row in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
