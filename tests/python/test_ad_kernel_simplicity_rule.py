import pytest

import taichi as ti


@ti.test()
def test_kernel_simplicity_rule():
    x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    arr = ti.field(dtype=ti.f32, shape=(2), needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def ad_mixed_usage_1():
        loss[None] += ti.sin(x[None])
        for i in arr:
            for j in range(2):
                for k in range(2):
                    for l in range(3):
                        loss[None] += ti.sin(x[None])

    @ti.kernel
    def ad_mixed_usage_2():
        for i in arr:
            for j in range(2):
                for k in range(2):
                    loss[None] += ti.sin(x[None])
                    for l in range(3):
                        loss[None] += ti.sin(x[None])

    @ti.kernel
    def ad_multiple_for_loops_1():
        for i in arr:
            for j in range(2):
                for k in range(2):
                    for l in range(3):
                        loss[None] += ti.sin(x[None])
            for j in range(2):
                for k in range(2):
                    for l in range(3):
                        loss[None] += ti.sin(x[None])

    @ti.kernel
    def ad_multiple_for_loops_2():
        for i in arr:
            for j in range(2):
                for k in range(2):
                    for l in range(3):
                        loss[None] += ti.sin(x[None])
                for j in range(2):
                    for k in range(2):
                        for l in range(3):
                            loss[None] += ti.sin(x[None])

    x[None] = 3.14
    with pytest.raises(
            RuntimeError,
            match="Invalid program input for autodiff: "
            "Mixed usage of for-loop and a statement without looping. \n"
            "Please check the documentation "
            "for the \"Kernel Simplicity Rule\" \"differentiable_task4\":\n"
            "https://docs.taichi.graphics/lang/articles/advanced/"
            "differentiable_programming#kernel-simplicity-rule"):
        with ti.Tape(loss=loss):
            ad_mixed_usage_1()
        with ti.Tape(loss=loss):
            ad_mixed_usage_2()

    with pytest.raises(
            RuntimeError,
            match="Invalid program input for autodiff: "
            "The outer for-loop contains more than one for-loops. \n"
            "Please check the documentation "
            "for the \"Kernel Simplicity Rule\" \"differentiable_task3\":\n"
            "https://docs.taichi.graphics/lang/articles/advanced/"
            "differentiable_programming#kernel-simplicity-rule"):
        with ti.Tape(loss=loss):
            ad_multiple_for_loops_1()
        with ti.Tape(loss=loss):
            ad_multiple_for_loops_2()
