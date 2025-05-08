import pytest
import os
from os.path import join
import subprocess

from tests import test_utils


def load_cpp_example_tests():
    files = [file for file in os.listdir("build") if file.startswith("cpp_examples_")]
    filepaths = [join("build", file) for file in files]
    return filepaths


@test_utils.test()
def test_exist_cpp_example_tests():
    print(os.getcwd())
    filepaths = load_cpp_example_tests()
    assert len(filepaths) > 0, "No cpp examples found in build directory"


@pytest.mark.parametrize("filepath", load_cpp_example_tests())
@test_utils.test()
def test_cpp_example(filepath: str) -> None:
    subprocess.check_output(filepath)
