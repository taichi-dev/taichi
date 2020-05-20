# Copyright (c) 2017 The Taichi Authors
# Use of this software is governed by the LICENSE file.

import os


def clear_directory_with_suffix(directory, suffix):
    files = os.listdir(directory)
    assert suffix[0] != '.', "No '.' needed."
    for f in files:
        if f.endswith('.' + suffix):
            os.remove(os.path.join(directory, f))
