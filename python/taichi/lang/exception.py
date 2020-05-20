# Copyright (c) 2020 The Taichi Authors
# Use of this software is governed by the LICENSE file.


class TaichiSyntaxError(Exception):
    def __init__(self, *args):
        super().__init__(*args)
