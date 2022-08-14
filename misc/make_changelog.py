# Usage: make_changelog.py --ver origin/master --save

import argparse
import json
import os
import re
import sys

from git import Repo


def load_pr_tags():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(this_dir, 'prtags.json')
    details = {}
    with open(json_path) as f:
        details = json.load(f)
    details['release'] = ''
    return details


def find_latest_tag_commit(tags):
    for tag in reversed(tags):
        s = re.match(r'v\s*([\d.]+)', tag.name)
        print(f'Latest version tag is: {tag.name}')
        if s is not None:
            return tag.commit


def main(ver=None, repo_dir='.'):
    g = Repo(repo_dir)
    g.tags.sort(key=lambda x: x.commit.committed_date, reverse=True)

    # We need to find out the latest common commit among base and ver,
    # everything after this commit should be listed in the changelog.

    base_commit = find_latest_tag_commit(g.tags)
    commits_in_base_tag = list(g.iter_commits(base_commit, max_count=200))
    commits = list(g.iter_commits(ver, max_count=200))
    begin, end = -1, 0

    def format(c):
        return f'{c.summary} (by **{c.author}**)'

    notable_changes = {}
    all_changes = []

    details = load_pr_tags()

    for i, c in enumerate(commits):
        s = format(c)
        if c in commits_in_base_tag and i > 0:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver")
    parser.add_argument("--repo_dir", type=str, default='.')
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()
    res = main(args.ver, args.repo_dir)
    if args.save:
        with open('./python/taichi/CHANGELOG.md', 'w', encoding='utf-8') as f:
            f.write(res)
    print(res)
