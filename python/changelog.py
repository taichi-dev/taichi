import argparse
import json
import os
import sys

sys.path.insert(0, './taichi')
import make_changelog
from git import Repo


def changelog(arguments: list = sys.argv[2:]):
    """Display changelog of current version"""
    parser = argparse.ArgumentParser(
        prog='ti changelog',
        description="Display changelog of current version")
    parser.add_argument(
        'version',
        nargs='?',
        type=str,
        default='master',
        help="A version (tag) that git can use to compare diff with")
    parser.add_argument(
        '-s',
        '--save',
        action='store_true',
        help="Save changelog to CHANGELOG.md instead of print to stdout")
    args = parser.parse_args(arguments)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = make_changelog.main(args.version, repo_dir)
    if args.save:
        changelog_md = os.path.join(repo_dir, 'CHANGELOG.md')
        print(changelog_md)
        with open(changelog_md, 'w') as f:
            f.write(res)
    else:
        print(res)


if __name__ == '__main__':
    changelog(sys.argv[1:])
