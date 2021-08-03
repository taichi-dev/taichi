import os
import sys

from git import Repo

commits = list(Repo('.').iter_commits('master'))

authors = {}
notable = {}
changelog = {}

for i, c in enumerate(commits):
    s = c.summary

    tags = []
    while s[0] == '[':
        r = s.find(']')
        tag = s[1:r]
        tags.append(tag)
        s = s[r + 1:]
        s = s.strip()

    for tag in tags:
        if tag[0].isupper():
            tag = tag.lower()
            notable[tag] = notable.get(tag, 0) + 1
        changelog[tag] = changelog.get(tag, 0) + 1

    a = str(c.author).split(' <')[0]
    authors[a] = authors.get(a, 0) + 1


def print_table(tab, name=''):
    print('')
    print(f' | {name} | counts |')
    print(' | :--- | ---: |')
    tab = sorted(tab.items(), key=lambda x: x[1], reverse=True)
    for a, n in tab:
        print(f' | {a} | {n} |')
    return tab


print('Authors:')
print_table(authors, 'author')
print('')
print('Highlights:')
print_table(notable, 'tag')
print('')
print('Full changelog:')
print_table(changelog, 'tag')
