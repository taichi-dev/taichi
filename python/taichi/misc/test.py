# Copyright (c) 2020 The Taichi Authors
# Use of this software is governed by the LICENSE file.

import taichi as ti


def approx(*args, **kwargs):
    from pytest import approx
    if ti.cfg.arch == ti.opengl:
        kwargs['rel'] = max(kwargs.get('rel', 1e-6), 1e-3)
        return approx(*args, **kwargs)
    else:
        return approx(*args, **kwargs)
