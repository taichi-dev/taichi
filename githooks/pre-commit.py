#!/usr/bin/env python3

import os
import sys
import taichi as tc
from git import Repo
from yapf.yapflib.yapf_api import FormatFile

repo = Repo(tc.get_repo_directory())

print('* Formatting code', end='')
for item in repo.index.diff('HEAD'):
    fn = os.path.join(tc.get_repo_directory(), item.a_path)
    print(end='.')
    if fn.endswith('.py'):
        FormatFile(fn, in_place=True, style_config=os.path.join(tc.get_repo_directory(), '.style.yapf'))
    if fn.endswith('.cpp'):
        os.system('clang-format -i -style=file {}'.format(fn))

print('* Done!')
