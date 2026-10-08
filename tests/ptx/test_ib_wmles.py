"""Opt-in, byte-for-byte CUDA compiler regression using pinned solver sources."""

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


SOLVER_COMMIT = "d0e815c559afde3c41e16ecdd435f8b1b137bd68"
HERE = Path(__file__).resolve().parent
GOLD = HERE / "gold" / "ib_wmles"
OPTIMIZATIONS = (
    "TI_CFG_COMPACT_REACHING",
    "TI_CFG_COMPACT_LIVE",
    "TI_CFG_PRECOMPUTED_KILLS",
    "TI_CSE_INDEXED_USERS",
)
VERIFICATION = ("TI_CFG_VERIFY_REACHING", "TI_CFG_VERIFY_LIVE", "TI_CSE_VERIFY_USERS")


@pytest.fixture(scope="session")
def solver_source(request, tmp_path_factory):
    repo = request.config.getoption("--simfinity-repo")
    if not repo:
        pytest.skip("requires --simfinity-repo, CUDA, and the solver's Python dependencies")
    destination = tmp_path_factory.mktemp("simfinity-development")
    archive = subprocess.check_output(["git", "-C", repo, "archive", SOLVER_COMMIT, "apps/solver"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter="data")
    return destination / "apps/solver"


@pytest.mark.parametrize("method", ["advection", "viscosity", "heat_conduction"])
def test_ib_wmles_ptx(method, solver_source, request, tmp_path):
    mode = request.config.getoption("--ptx-mode")
    record = request.config.getoption("--record-ptx-gold")
    if record and mode != "reference":
        pytest.fail("Gold may only be recorded with --ptx-mode=reference on an unpatched build")
    env = dict(os.environ)
    # Reject inherited compiler-policy overrides, including flags unrelated to
    # this patch. Baseline and candidate must use the same default optimizations.
    for key in list(env):
        if key.startswith("TI_"):
            del env[key]
    env.update({key: str(int(mode != "reference")) for key in OPTIMIZATIONS})
    env.update({key: str(int(mode == "verify")) for key in VERIFICATION})
    env.update(TI_OFFLINE_CACHE="0", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    with (tmp_path / "compile.log").open("w") as log:
        process = subprocess.run(
            [
                sys.executable,
                str(HERE / "compile_ib_wmles.py"),
                "--solver-root",
                str(solver_source),
                "--method",
                method,
            ],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert process.returncode == 0, (tmp_path / "compile.log").read_text()[-12000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["compile_only"] is True
    assert len(result["records"]) == (3 if method == "heat_conduction" else 2)
    destination = GOLD / method
    if record:
        assert not result["compiler_has_patch"], "Record gold before applying the compiler patch"
        destination.mkdir(parents=True, exist_ok=True)
        result["solver_commit"] = SOLVER_COMMIT
        for item in result["records"]:
            data = (tmp_path / item["file"]).read_bytes()
            (destination / (item["file"] + ".gz")).write_bytes(gzip.compress(data, mtime=0))
        (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        reference = json.loads((destination / "manifest.json").read_text())
        assert reference["solver_commit"] == SOLVER_COMMIT
        assert result["llvm_version"] == reference["llvm_version"], "Gold requires the same LLVM version"
        assert result["cuda_compute_capability"] == reference["cuda_compute_capability"], "PTX target changed"
        assert [item["body"] for item in result["records"]] == [item["body"] for item in reference["records"]]
        for actual, expected in zip(result["records"], reference["records"]):
            gold = gzip.decompress((destination / (expected["file"] + ".gz")).read_bytes())
            assert hashlib.sha256(gold).hexdigest() == expected["sha256"], "Corrupt gold PTX"
            generated = (tmp_path / actual["file"]).read_bytes()
            assert generated == gold, (
                f"PTX changed for {method}/{actual['body']}; actual: {tmp_path / actual['file']}; "
                f"expected: {destination / (expected['file'] + '.gz')}"
            )
