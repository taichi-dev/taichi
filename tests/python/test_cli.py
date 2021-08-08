import argparse
import copy
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from taichi.main import TaichiMain

import taichi as ti


@contextmanager
def patch_sys_argv_helper(custom_argv: list):
    """Temporarily patch sys.argv for testing."""
    try:
        cached_argv = copy.deepcopy(sys.argv)
        sys.argv = custom_argv
        yield sys.argv
    finally:
        sys.argv = cached_argv


def test_cli_exit_one_with_no_command_provided():
    with patch_sys_argv_helper(["ti"]):
        cli = TaichiMain(test_mode=True)
        assert cli() == 1


def test_cli_exit_one_with_bogus_command_provided():
    with patch_sys_argv_helper(["ti", "bogus-command-not-registered-yet"]):
        cli = TaichiMain(test_mode=True)
        assert cli() == 1


def test_cli_can_dispatch_commands_to_methods_correctly():
    with patch_sys_argv_helper(
        ["ti", "example", "bogus_example_name_for_test"]):
        with patch.object(TaichiMain, 'example',
                          return_value=None) as mock_method:
            cli = TaichiMain(test_mode=False)
            cli()
            mock_method.assert_called_once_with(
                ["bogus_example_name_for_test"])


def test_cli_example():
    with patch_sys_argv_helper(["ti", "example", "minimal"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal"

    with patch_sys_argv_helper(["ti", "example", "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal"

    with patch_sys_argv_helper(["ti", "example", "-s",
                                "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal" and args.save == True

    with patch_sys_argv_helper(["ti", "example", "-p",
                                "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal" and args.print == True

    with patch_sys_argv_helper(["ti", "example", "-P",
                                "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal" and args.pretty_print == True


def test_cli_gif():
    with patch_sys_argv_helper(["ti", "gif", "-i", "video.mp4", "-f",
                                "30"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.input_file == "video.mp4"
        assert args.framerate == 30
        assert args.output_file == "video.gif"

    with patch_sys_argv_helper(["ti", "gif", "-i", "video.mp3", "-f",
                                "30"]) as custom_argv:
        with pytest.raises(SystemExit) as pytest_wrapped_err:
            cli = TaichiMain(test_mode=True)
            args = cli()
            assert pytest_wrapped_err.__context__.type == argparse.ArgumentTypeError


def test_cli_video_speed():
    with patch_sys_argv_helper(
        ["ti", "video_speed", "-i", "video.mp4", "-s", "2.0"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.input_file == "video.mp4"
        assert args.speed == 2.0
        assert args.output_file == "video-sped.mp4"

    with patch_sys_argv_helper(
        ["ti", "video_speed", "-i", "video.mp3", "-s", "2.0"]) as custom_argv:
        with pytest.raises(SystemExit) as pytest_wrapped_err:
            cli = TaichiMain(test_mode=True)
            args = cli()
            assert pytest_wrapped_err.__context__.type == argparse.ArgumentTypeError


def test_cli_video_crop():
    with patch_sys_argv_helper([
            "ti", "video_crop", "-i", "video.mp4", "--x1", "10.0", "--x2",
            "20.0", "--y1", "10.0", "--y2", "20.0"
    ]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.input_file == "video.mp4"
        assert args.x_begin == 10.0
        assert args.x_end == 20.0
        assert args.y_begin == 10.0
        assert args.y_end == 20.0
        assert args.output_file == "video-cropped.mp4"

    with patch_sys_argv_helper([
            "ti", "video_crop", "-i", "video.mp3", "--x1", "10.0", "--x2",
            "20.0", "--y1", "10.0", "--y2", "20.0"
    ]) as custom_argv:
        with pytest.raises(SystemExit) as pytest_wrapped_err:
            cli = TaichiMain(test_mode=True)
            args = cli()
            assert pytest_wrapped_err.__context__.type == argparse.ArgumentTypeError


def test_cli_video_scale():
    with patch_sys_argv_helper(
        ["ti", "video_scale", "-i", "video.mp4", "-w", "1.2"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.input_file == "video.mp4"
        assert args.ratio_width == 1.2
        assert args.ratio_height == 1.2
        assert args.output_file == "video-scaled.mp4"

    with patch_sys_argv_helper([
            "ti", "video_scale", "-i", "video.mp4", "-w", "1.2",
            "--ratio-height", "1.5"
    ]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.input_file == "video.mp4"
        assert args.ratio_width == 1.2
        assert args.ratio_height == 1.5
        assert args.output_file == "video-scaled.mp4"

    with patch_sys_argv_helper([
            "ti", "video_scale", "-i", "video.mp3", "-w", "1.2",
            "--ratio-height", "1.5"
    ]) as custom_argv:
        with pytest.raises(SystemExit) as pytest_wrapped_err:
            cli = TaichiMain(test_mode=True)
            args = cli()
            assert pytest_wrapped_err.__context__.type == argparse.ArgumentTypeError


def test_cli_video():
    with patch_sys_argv_helper(
        ["ti", "video", "image.gif", "-o", "video.mp4", "-f",
         "30"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.inputs == ["image.gif"]
        assert args.framerate == 30
        assert isinstance(args.output_file, Path)
        assert args.output_file.name == "video.mp4"

    with patch_sys_argv_helper(["ti", "video", "-o", "video.mp4", "-f",
                                "30"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert isinstance(args.inputs, list)
        assert args.framerate == 30
        assert isinstance(args.output_file, Path)
        assert args.output_file.name == "video.mp4"


def test_cli_format():
    with patch_sys_argv_helper(["ti", "format", "51fff7af"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.diff == "51fff7af"


def test_cli_regression():
    with patch_sys_argv_helper(["ti", "regression", "a.py", "b.py",
                                "-g"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.files == ["a.py", "b.py"]
        assert args.gui == True


def test_cli_benchmark():
    with patch_sys_argv_helper(
        ["ti", "benchmark", "a.py", "b.py", "-T", "-v", "-r2",
         "-t4"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.files == ["a.py", "b.py"]
        assert args.tprt == True
        assert args.verbose == True
        assert args.rerun == "2"
        assert args.threads == "4"


def test_cli_test():
    with patch_sys_argv_helper(
        ["ti", "test", "cli", "atomic", "-c", "-v", "-r2",
         "-t4"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.files == ["cli", "atomic"]
        assert args.cpp == True
        assert args.verbose == True
        assert args.rerun == "2"
        assert args.threads == "4"

    with patch_sys_argv_helper(
        ["ti", "test", "cli", "atomic", "-c", "-v", "-r2",
         "-t4"]) as custom_argv:
        with patch.object(TaichiMain, 'test', return_value=1) as mock_method:
            cli = TaichiMain(test_mode=False)
            return_code = cli()
            assert return_code == 1


def test_cli_debug():
    with patch_sys_argv_helper(["ti", "debug", "a.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.filename == "a.py"


def test_cli_run():
    with patch_sys_argv_helper(["ti", "run", "a.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.filename == "a.py"


def test_cli_task():
    with patch_sys_argv_helper(["ti", "task", "test_task", "arg1",
                                "arg2"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.taskname == "test_task"
        assert args.taskargs == ["arg1", "arg2"]
