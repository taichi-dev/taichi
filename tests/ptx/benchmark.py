"""Compare two built Taichi packages against fixed IB-WMLES PTX gold.

Run with the solver's Python environment. Package paths are PYTHONPATH roots,
each containing a taichi package with its native extension and runtime bitcode.
Both builds use the optimized compiler mode. Every sample runs in fresh child
processes with offline caching disabled; candidate/control order alternates.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pythonpath", type=Path, required=True)
    parser.add_argument("--candidate-pythonpath", type=Path, required=True)
    parser.add_argument("--simfinity-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tests = Path(__file__).resolve().parent
    samples = []
    packages = {
        "baseline": args.baseline_pythonpath.resolve(),
        "candidate": args.candidate_pythonpath.resolve(),
    }
    for repeat in range(args.repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            destination = output / f"{repeat}-{variant}"
            env = dict(os.environ, PYTHONPATH=str(packages[variant]))
            command = [
                sys.executable, "-m", "pytest", str(tests), "-v",
                "--simfinity-repo", str(args.simfinity_repo.resolve()),
                "--ptx-mode", "optimized", "--basetemp", str(destination),
            ]
            with (output / f"{repeat}-{variant}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            paths = sorted({path.resolve() for path in destination.glob("test_*/result.json")})
            results = [json.loads(path.read_text()) for path in paths]
            assert len(results) == 3
            assert {item["method"] for item in results} == {
                "advection", "viscosity", "heat_conduction"
            }
            sample = {"repeat": repeat, "variant": variant, "results": results}
            samples.append(sample)
            (output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
            seconds = sum(record["seconds"] for result in results for record in result["records"])
            print(f"{repeat} {variant}: {seconds:.3f} s; all seven PTX modules identical", flush=True)

    summary = {}
    for method in ("advection", "viscosity", "heat_conduction", "total"):
        row = {}
        for variant in packages:
            values = [
                sum(record["seconds"] for result in sample["results"]
                    if method == "total" or result["method"] == method
                    for record in result["records"])
                for sample in samples if sample["variant"] == variant
            ]
            row[variant] = {
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
        row["speedup"] = row["baseline"]["median_seconds"] / row["candidate"]["median_seconds"]
        summary[method] = row
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
