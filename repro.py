import numpy as np
import torch

import taichi as ti

ti.init(ti.cuda, log_level=ti.TRACE)


def test_shape_vector():
    n = 3
    x = ti.Vector.field(3, ti.f32, shape=(n, n))
    print('============first time begin============', flush=True)
    X = x.to_torch()
    print('============first time done============', flush=True)

    X1 = x.to_torch()
    print('============second time done============', flush=True)

    # print(x.to_numpy())
    print(X1)

    assert (X == X1).all()


test_shape_vector()
