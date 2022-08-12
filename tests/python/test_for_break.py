import taichi as ti
from tests import test_utils


@test_utils.test()
def test_for_break():
    x = ti.field(ti.i32)
    N, M = 4, 4
    ti.root.dense(ti.ij, (N, M)).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            for j in range(M):
                if j > i:
                    break
                x[i, j] = 100 * i + j

    func()
    for i in range(N):
        for j in range(M):
            if j > i:
                assert x[i, j] == 0
            else:
                assert x[i, j] == 100 * i + j


@test_utils.test()
def test_for_break2():
    x = ti.field(ti.i32)
    N, M = 8, 8
    ti.root.dense(ti.ij, (N, M)).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            for j in range(M):
                x[i, j] = 100 * i + j
                if j > i:
                    break

    func()
    for i in range(N):
        for j in range(M):
            if j > i + 1:
                assert x[i, j] == 0
            else:
                assert x[i, j] == 100 * i + j


@test_utils.test(exclude=ti.vulkan)
def test_for_break3():
    x = ti.field(ti.i32)
    N, M = 8, 8
    ti.root.dense(ti.ij, (N, M)).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            for j in range(i, M - i):
                if i == 0:
                    break
                x[i, j] = 100 * i + j

    func()
    for i in range(N):
        for j in range(M):
            if j < i or j >= M - i or i == 0:
                assert x[i, j] == 0
            else:
                assert x[i, j] == 100 * i + j


@test_utils.test()
def test_for_break_complex():
    x = ti.field(ti.i32)
    N, M = 16, 32
    ti.root.dense(ti.ij, (N, M)).place(x)

    @ti.kernel
    def func():
        for i in range(1, N):
            for j in range(3, M):
                if j > i:
                    break
                x[i, j] = 100 * i + j

    func()
    for i in range(N):
        for j in range(M):
            if i < 1 or j < 3 or j > i:
                assert x[i, j] == 0
            else:
                assert x[i, j] == 100 * i + j


@test_utils.test()
def test_serial_for_with_break_and_continue():
    @ti.kernel
    def test_kernel() -> ti.i32:
        stop = 0
        sum = 0
        ti.loop_config(serialize=True)
        for i in range(10):
            if i % 2 == 0:
                continue
            sum += i
            if stop:
                break
        return sum

    assert test_kernel() == 25
