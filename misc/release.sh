#!/bin/bash
set -e

git tag ${tag?no tag specified}
git push --tags -u origin master
git checkout stable
git merge master
git push -u origin stable
clear
python misc/make_changelog.py $tag
