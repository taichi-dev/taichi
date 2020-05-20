# Copyright (c) 2017 The Taichi Authors
# Use of this software is governed by the LICENSE file.

from taichi.core import unit


@unit('task')
class Task:
    def run(self, *args):
        return self.c.run(args)
