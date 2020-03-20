import glob
import os

line_start = '/*******************************************************************************'
line_end = '*******************************************************************************/'

copyright_content = \
'''
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
'''

files = list(glob.iglob('src/**/*', recursive=True)) + list(
    glob.iglob('include/**/*', recursive=True))

for fn in files:
    if os.path.isdir(fn):
        continue

    with open(fn, 'r') as f:
        lines = f.readlines()

    print(fn)

    start_lines = [i for (i, l) in enumerate(lines) if l.strip() == line_start]
    end_lines = [i for (i, l) in enumerate(lines) if l.strip() == line_end]

    assert (len(start_lines) == 1)
    assert (len(end_lines) == 1)
    start_lines = start_lines[0]
    end_lines = end_lines[0]
    assert (start_lines == 0)

    if end_lines == 1:
        continue

    assert (7 <= end_lines < 9)

    lines = lines[end_lines + 1:]

    lines = [line_start] + [copyright_content] + [line_end + '\n'] + lines

    #print(''.join(lines))

    with open(fn, 'w') as f:
        f.writelines(lines)
