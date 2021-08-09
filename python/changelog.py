import os
import sys
import argparse
import json

from git import Repo


def changelog(arguments: list = sys.argv[2:]):
    """Display changelog of current version"""
    parser = argparse.ArgumentParser(
        prog='ti changelog', description="Display changelog of current version")
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
        help="Save changelog to CHANGELOG.md instead of print to stdout"
    )
    args = parser.parse_args(arguments)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = make_changelog(args.version, repo_dir)
    if args.save:
        changelog_md = os.path.join(repo_dir, 'CHANGELOG.md')
        print(changelog_md)
        with open(changelog_md, 'w') as f:
            f.write(res)
    else:
        print(res)

def load_pr_tags():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(this_dir, '../misc/prtags.json')
    details = {}
    with open(json_path) as f:
        details = json.load(f)
    details['release'] = ''
    return details


def make_changelog(ver='master', repo_dir='.'):
    g = Repo(repo_dir)
    commits = list(g.iter_commits(ver, max_count=200))
    begin, end = -1, 0

    def format(c):
        return f'{c.summary} (by **{c.author}**)'

    notable_changes = {}
    all_changes = []

    details = load_pr_tags()

    for i, c in enumerate(commits):
        s = format(c)
        if s.startswith('[release]'):
            if i == 0:
                continue
            else:
                break

        tags = []
        while s[0] == '[':
            r = s.find(']')
            tag = s[1:r]
            tags.append(tag)
            s = s[r + 1:]
            s = s.strip()

        for tag in tags:
            if tag.lower() in details:
                if details[tag.lower()] == '':
                    # E.g. 'release' does not need to appear in the change log
                    continue
                if tag[0].isupper():
                    tag = tag.lower()
                    if tag not in notable_changes:
                        notable_changes[tag] = []
                    notable_changes[tag].append(s)
            else:
                print(
                    f'** Warning: tag {tag.lower()} undefined in the "details" dict. Please include the tag into "details", unless the tag is a typo.'
                )

        all_changes.append(format(c))

    res = 'Highlights:\n'
    for tag in sorted(notable_changes.keys()):
        res += f'   - **{details[tag]}**\n'
        for item in notable_changes[tag]:
            res += f'      - {item}\n'

    res += '\nFull changelog:\n'
    for c in all_changes:
        res += f'   - {c}\n'

    return res


if __name__ == '__main__':
    changelog(sys.argv[1:])
