# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# -- third party --
from pytest import ExceptionInfo

# -- own --

# -- code --
BASE = Path(__file__).parent.resolve()


def pytest_collect_file(parent, file_path):
    if file_path.name == "cpptests.yaml":
        return CPPTestsFile.from_parent(parent, path=file_path)


class CPPTestsFile(pytest.File):
    def collect(self):
        cpptests = yaml.safe_load(open(self.path).read())

        for suite in cpptests:
            sname = suite["name"]
            binary = suite["binary"]
            if platform.system() == "Windows":
                binary += ".exe"

            binary = BASE / binary
            if not binary.exists():
                return

            seen = set()

            # defined in the yaml file
            for test in suite["tests"]:
                seen.add(test["test"])
                name = f'{sname} - {test["test"]}'
                item = CPPTestItem.from_parent(
                    self,
                    name=name,
                    binary=binary,
                    test=test["test"],
                    script=test.get("script"),
                    args=test.get("args"),
                )
                for m in test.get("markers", []):
                    item.add_marker(getattr(pytest.mark, m))

                yield item

            # the rest, with default configuration
            for tname in self.list_all_tests(binary):
                if tname not in seen:
                    name = f"{sname} - {tname}"
                    yield CPPTestItem.from_parent(
                        self,
                        name=name,
                        binary=binary,
                        test=tname,
                    )

    def list_all_tests(self, binary):
        proc = subprocess.Popen([str(binary), "--gtest_list_tests"], stdout=subprocess.PIPE)
        out, _ = proc.communicate()

        lst = []

        lines = list(reversed(out.decode().splitlines()))

        # skip junk lines
        while lines:
            l = lines.pop().strip()
            if l.endswith("."):
                break
        else:
            raise Exception("Unexpected output")

        mod = l
        while lines:
            l = lines.pop().rstrip()
            if l.startswith("  ") and not l.endswith("."):
                l = l.split("#", 2)[0].strip()
                lst.append(f"{mod}{l}")
                continue
            elif l.endswith("."):
                mod = l.strip()
            else:
                raise Exception(f"Unexpected line: {l}")

        return lst


class CPPTestItem(pytest.Item):
    def __init__(self, *, binary, test, script=None, args=None, **kwargs):
        super().__init__(**kwargs)
        self.binary = binary
        self.test = test
        self.script = script
        self.args = args

    def runtest(self):
        import taichi as ti

        ti_lib_dir = Path(ti.__path__[0]) / "_lib" / "runtime"

        with tempfile.TemporaryDirectory(prefix="ti-cpp-tests-") as tmpdir:
            try:
                env = os.environ.copy()
                env.update(
                    {
                        "TI_DEVICE_MEMORY_GB": "0.5",
                        "TI_LIB_DIR": str(ti_lib_dir),
                        "TAICHI_AOT_FOLDER_PATH": tmpdir,
                    }
                )
                if self.script:
                    retcode = subprocess.call(
                        f"{sys.executable} {self.script} {self.args}",
                        shell=True,
                        cwd=str(BASE),
                        env=env,
                    )

                    retcode and pytest.fail(f"{self.script} {self.args} reported failure, exit code {retcode}")

                retcode = subprocess.call(
                    f"{self.binary} --gtest_filter={self.test}",
                    shell=True,
                    cwd=str(BASE),
                    env=env,
                )

                retcode and pytest.fail(f"C++ part reported failure, exit code {retcode}")

            except Exception:
                excinfo = sys.exc_info()
                raise CPPTestException(excinfo)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, CPPTestException):
            return super().repr_failure(ExceptionInfo(excinfo.value.excinfo))

        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, self.name


class CPPTestException(Exception):
    """Custom exception for error reporting."""

    def __init__(self, excinfo):
        self.excinfo = excinfo
