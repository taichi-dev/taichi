#!/usr/bin/env python3
# Copyright (c) 2020 The Taichi Authors. All rights reserved.
# Use of this software is governed by the LICENSE file.
"""
Open each source file and add a copyright notice if it is missing.
Could be faster with multiprocessing or async/await but this is
just an one-off script on a fairly-sized project so it's unnecessary.
"""

import argparse
import os
import re
import time
import subprocess
import sys
from datetime import datetime
from enum import Enum
from typing import List

# Sync with implementation
# Do not use color sequences or hyperlink sequences so as to be portable on Windows.
DOC_STRING = """Copyright notice inserter.

Arguments:

paths       For each file, it will be visited; for each directory, files under
            that tree will be visited.

--exts      Comma-separated extension names to restrict the files that to be
            picked up, e.g. "cpp", "cpp,py,sh". If not given, all files on the
            traversal paths with recognized extension names will be picked up.

--dry-run   Do not modify the files (for development).

--docs      Print this long documentation and exits.

Effects:

The scripts checks a file and modifies it if necessary.
    (1) If a copyright notice is absent, it inserts it at the start.
        As an exception, if there is a sharp-bang (#!) line at the first line,
        the new notice is inserted after it.
    (2) If a copyright notice is present, but is in a specific wrong format, it
        modifies the notice: https://github.com/taichi-dev/taichi/issues/1024

Examples:

{file} .
{file} taichi/common/core.cpp
{file} benchmarks cmake docs examples misc python taichi tests
{file} --exts "cpp,py" benchmarks cmake docs examples misc python taichi tests""".format(
    file=os.path.relpath(__file__))


class CommentStyle(Enum):
    C_STYLE = 1  # /* .. */
    CPP_STYLE = 2  # // .. => seems to be used in *.cu only
    PY_STYLE = 3  # #..


class FileActionResult(Enum):
    INTACT = 1
    INSERTED_NOTICE = 2
    MODIFIED_NOTICE = 3


# Sync with DOC_STRING
FILE_EXT_TO_COMMENT_STYLES = {
    ".h": CommentStyle.C_STYLE,
    ".c": CommentStyle.C_STYLE,
    ".cc": CommentStyle.C_STYLE,
    ".cpp": CommentStyle.C_STYLE,
    # Though CUDA is on top of C but existing files seem
    # to prefer "//" comments.
    ".cu": CommentStyle.CPP_STYLE,
    ".py": CommentStyle.PY_STYLE,
    ".sh": CommentStyle.PY_STYLE,
}


def get_ctime_year(filepath: str) -> str:
    """
    Get file creation time and return the year as a stringified integer.
    Because git-clone overwrites ctime, we need to retrieve it
    using git-log.
    """
    # %as: author date as a YYYY-MM-DD string
    command = "git --no-pager log --reverse --format=\"%as\" {}".format(
        filepath)
    try:
        out = subprocess.check_output(command.split())
    except subprocess.CalledProcessError as e:
        sys.exit("%s error: %s" % (command, e))
    assert (type(out) == bytes)
    initial_commit_time = out.decode().strip('"').strip("'").split('\n', 1)[0]
    return initial_commit_time.split('-')[0]  # str


def make_notice(comment_style: CommentStyle, ctime_year: str) -> List[str]:
    """
    Returns the notice message as list of strings.
    NOTE Each line should end with a newline character.
    """
    lines = []
    if comment_style == CommentStyle.C_STYLE:
        lines.append("/*" + "*" * 78 + "\n")
        line_start = "    "
    elif comment_style == CommentStyle.CPP_STYLE:
        line_start = "//"
    elif comment_style == CommentStyle.PY_STYLE:
        line_start = "#"
    lines.append(
        "{0} Copyright (c) {1} The Taichi Authors. All rights reserved.\n".
        format(line_start, ctime_year))
    lines.append(
        "{0} Use of this software is governed by the LICENSE file.\n".format(
            line_start))
    if comment_style == CommentStyle.C_STYLE:
        lines.append("*" * 78 + "*/\n")
    lines.append("\n")
    return lines


# Sync with DOC_STRING
COPYRIGHT_NOTICE_REGEX = re.compile(r"copyright.+taichi")
COPYRIGHT_INCORRECT_REGEX = re.compile(r"copyright.+taichi.+(20\d\d)-")


def check_and_modify(filepath: str, comment_style: CommentStyle,
                     dry_run: bool) -> FileActionResult:
    """
    Effects: see DOC_STRING
    """
    new_header_lines, body_lines = [], []
    with open(filepath, 'r') as f:
        body_lines = f.readlines()
    existent_notice_match, incorrect_notice_match = None, None
    sharp_bang_line = None
    for i in range(8):
        if i >= len(body_lines):
            continue
        line = body_lines[i]
        if i == 0 and line.startswith('#!'):
            sharp_bang_line = line
        line_lower = line.lower()
        existent_notice_match = COPYRIGHT_NOTICE_REGEX.search(line_lower)
        if existent_notice_match:  # Notice exists...
            incorrect_notice_match = COPYRIGHT_INCORRECT_REGEX.search(
                line_lower)
            if not incorrect_notice_match:  # ...and is not caught by the format checker
                return FileActionResult.INTACT
            else:  # ...but is caught by the format checker
                to_replace_line_index = i
                break
    if not existent_notice_match:
        assert (not incorrect_notice_match)
        # Notice missing; now we need to insert a notice.
        new_header_lines = make_notice(comment_style, get_ctime_year(filepath))
        if sharp_bang_line:
            new_header_lines = [sharp_bang_line] + new_header_lines
            body_lines = body_lines[1:]  # Remove the original #! line
        return_state = FileActionResult.INSERTED_NOTICE
    else:
        assert (incorrect_notice_match)
        # Notice exists but format is wrong; now we need to modify that notice.
        notice_match_start = existent_notice_match.start()
        year_1st = incorrect_notice_match.group(1)
        assert (year_1st)
        # This is how cs.chromium.org writes the notice, and I think the lawyers
        # should be confident :)
        correct_line = (" " * notice_match_start) \
            + "Copyright (c) %s The Taichi Authors. All rights reserved.\n" % year_1st
        body_lines[to_replace_line_index] = correct_line
        return_state = FileActionResult.MODIFIED_NOTICE
    if not dry_run:
        with open(filepath, 'w') as f:
            if new_header_lines:
                f.writelines(new_header_lines)
            f.writelines(body_lines)
    return return_state


class WorkStats:
    def __init__(self):
        self.opened_file_num = 0
        self.inserted_notice_file_num = 0
        self.modified_notice_file_num = 0


# Available on POSIX, Windows.
try:
    LINE_ELIDING = os.get_terminal_size().columns >= 72
except AttributeError:  # Maybe we want to run on niche platforms..
    LINE_ELIDING = False


PLAYFUL_BRAILLE = ["⠃", "⠆", "⠤", "⠰", "⠘", "⠉"]


def print_progress(stats: WorkStats, dry_run: bool):
    content = "{dots} Opened {opened}, {tense} insert notice: {insert}, {tense} modify notice: {modify}".format(
        opened=stats.opened_file_num,
        tense="will" if dry_run else "did",
        insert=stats.inserted_notice_file_num,
        modify=stats.modified_notice_file_num,
        dots=PLAYFUL_BRAILLE[stats.opened_file_num % len(PLAYFUL_BRAILLE)])
    # 1A, 2K: move cursor one line up and clear the entire line.
    sys.stdout.write(("\x1b[1A\x1b[2K" if LINE_ELIDING else "") + content + "\n")


def work_on_file(filepath: str, ext: str, stats: WorkStats, dry_run: bool):
    stats.opened_file_num += 1
    status = check_and_modify(filepath, FILE_EXT_TO_COMMENT_STYLES[ext], dry_run)
    if status == FileActionResult.INSERTED_NOTICE:
        stats.inserted_notice_file_num += 1
    elif status == FileActionResult.MODIFIED_NOTICE:
        stats.modified_notice_file_num += 1
    print_progress(stats, dry_run)
    return


def is_interested_ext(ext: str, selected_stripped_exts: List[str]) -> bool:
    """
    Returns whether we are interested in a file based on its extension name.
    NOTE ext contains a dot, e.g. ".cpp", ".sh", while selected_stripped_exts elements
         do not, e.g. [ "cpp", "sh" ]
    """
    return (ext in FILE_EXT_TO_COMMENT_STYLES) \
        and ((not selected_stripped_exts) or (ext.lstrip(".") in selected_stripped_exts))


def work(args):
    start_time = time.time()
    stats = WorkStats()
    picked_exts = [] if not args.exts else args.exts.split(",")
    print("Starting.")
    for path in args.paths:
        if os.path.isdir(path):
            for (dirpath, dirnames, filenames) in os.walk(path):
                for f in filenames:
                    ext = os.path.splitext(f)[-1]
                    if not is_interested_ext(ext, picked_exts):
                        continue
                    work_on_file(os.path.join(dirpath, f), ext, stats,
                                 args.dry_run)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[-1]
            if not is_interested_ext(ext, picked_exts):
                continue
            work_on_file(path, ext, stats, args.dry_run)
    print("Done in %.1f sec." % (time.time() - start_time))


def main():
    argparser = argparse.ArgumentParser(
        description=
        "Copyright notice inserter: insert if missing; modify if wrong.")
    argparser.add_argument("paths",
                           type=str,
                           nargs='*',
                           help="non-overlapping directories or files")
    argparser.add_argument(
        "-e",
        "--exts",
        metavar="E,..",
        type=str,
        default="",
        help="comma-separated file extensions; all if absent")
    argparser.add_argument("-d",
                           "--dry-run",
                           action="store_true",
                           help="(dev) do not modify files")
    argparser.add_argument("--docs",
                           action="store_true",
                           help="print long documentation")
    args = argparser.parse_args()
    if args.docs:
        print(DOC_STRING)
        return 0

    if len(args.paths) == 0:
        sys.exit("[Error] at least one path required")

    missing_dirs = [e for e in args.paths if not os.path.exists(e)]
    if missing_dirs:
        sys.exit("[Error] path not found: %s" % " ".join(missing_dirs))

    unhandled_exts = None if not args.exts else [
        e for e in args.exts.split(",")
        if ("." + e) not in FILE_EXT_TO_COMMENT_STYLES
    ]
    if unhandled_exts:
        sys.exit("[Error] unhandled extension names: %s" %
                 " ".join(unhandled_exts))
    return work(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
