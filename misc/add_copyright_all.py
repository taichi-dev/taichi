#!/usr/bin/env python3
# Copyright (c) 2020 The Taichi Authors
# Use of this software is governed by the LICENSE file.
"""
Open each source file and add a copyright notice if it is missing.
Could be faster with multiprocessing or async/await but this is
just an one-off script on a fairly-sized project so it's unnecessary.
"""

import argparse
import os
import time
import subprocess
import sys
from datetime import datetime
from enum import Enum


class CommentStyle(Enum):
    C_STYLE = 1  # /* .. */
    CPP_STYLE = 2  # // .. => seems to be used in *.cu only
    PY_STYLE = 3  # #..


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


def get_ctime_year(filepath):
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


def make_notice(comment_style, ctime_year):
    """
    Returns the notice message as list of strings.
    """
    lines = []
    if comment_style == CommentStyle.C_STYLE:
        lines.append(
            "/*******************************************************************************\n"
        )
        line_start = "    "
    elif comment_style == CommentStyle.CPP_STYLE:
        line_start = "//"
    elif comment_style == CommentStyle.PY_STYLE:
        line_start = "#"
    lines.append("{0} Copyright (c) {1} The Taichi Authors\n".format(
        line_start, ctime_year))
    lines.append(
        "{0} Use of this software is governed by the LICENSE file.\n".format(
            line_start))
    if comment_style == CommentStyle.C_STYLE:
        lines.append(
            "*******************************************************************************/\n"
        )
    lines.append("\n")
    return lines


def check_and_modify(filepath, comment_style, dry_run):
    """
    Returns True if the file was modified.
    """
    original_lines = []
    with open(filepath, 'r') as f:
        original_lines = f.readlines()
    for i in range(5):
        if i >= len(original_lines):
            continue
        line = original_lines[i]
        if "copyright" in line.lower():
            return False
    # Now we need to insert a notice.
    notice_lines = make_notice(comment_style, get_ctime_year(filepath))
    if dry_run:
        return True
    with open(filepath, 'w') as f:
        f.writelines(notice_lines)
        f.writelines(original_lines)
    return True


def print_progress(checked_num, modified_num, dry_run):
    content = "Opened {0}, {1} {2} {3}".format(
        checked_num, "modified" if not dry_run else "want to modify",
        modified_num, "." * (checked_num % 5))
    # 1A, 2K: move cursor one line up and clear the entire line.
    sys.stdout.write(("\x1b[1A\x1b[2K%s\n" % content))


def main():
    argparser = argparse.ArgumentParser("Copyright notice inserter")
    argparser.add_argument("root", type=str, help="recursion root path")
    argparser.add_argument("-d",
                           "--dry-run",
                           action="store_true",
                           help="do not modify files")
    args = argparser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit("[Error] directory not found: %s" % args.root)

    checked_file_num, modified_file_num = 0, 0
    start_time = time.time()
    print("Starting.")
    for (dirpath, dirnames, filenames) in os.walk(args.root):
        for f in filenames:
            ext = os.path.splitext(f)[-1]
            if ext not in FILE_EXT_TO_COMMENT_STYLES:
                continue
            filepath = os.path.join(dirpath, f)
            checked_file_num += 1
            print_progress(checked_file_num, modified_file_num, args.dry_run)
            if check_and_modify(filepath, FILE_EXT_TO_COMMENT_STYLES[ext],
                                args.dry_run):
                modified_file_num += 1
    print("Done in %.1f sec." % (time.time() - start_time))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
