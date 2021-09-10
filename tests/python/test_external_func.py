import pytest
import ctypes
import os

import taichi as ti


@ti.test(arch=[ti.cpu, ti.cuda], require=ti.extension.extfunc)
def test_so_and_bc():
    source_so = '''
    extern "C" {
        void add_and_mul(float a, float b, float *c, float *d, int *e) {
            *c = a + b;
            *d = a * b;
            *e = int(a * b + a);
        }
        void pow_int(int a, int b, int *c) {
            int ret = 1;
            for (int i = 0; i < b; i++)
                ret = ret * a;
            *c = ret;
        }
    }
    '''
    source_bc = '''
    extern "C" {
        void add_and_mul(float *a, float *b, float *c, float *d, int *e) {
            *c = (*a) + (*b);
            *d = (*a) * (*b);
            *e = int((*a) * (*b) + (*a));
        }
        void pow_int(int *a, int *b, int *c) {
            int ret = 1;
            for (int i = 0; i < (*b); i++)
                ret = ret * (*a);
            *c = ret;
        }
    }
    '''
    sb_so = ti.SourceBuilder(source_so, 'so')
    sb_bc = ti.SourceBuilder(source_bc, 'bc')
    @ti.kernel
    def func_so() -> ti.i32:
        a = 2.0
        b = 3.0
        c = 0.0
        d = 0.0
        e = 3
        sb_so.add_and_mul(args=(a, b), outputs=(c, d, e))
        p = 0
        sb_so.pow_int(args=(int(c + d), e), outputs=(p, ))
        return p
    @ti.kernel
    def func_bc() -> ti.i32:
        a = 2.0
        b = 3.0
        c = 0.0
        d = 0.0
        e = 3
        # sb_bc.add_and_mul(a, b, c, d, e)
        p = 0
        c_plus_d = int(c + d)
        sb_bc.pow_int(c_plus_d, e, p)
        return p
    assert func_so() == 11**8
    func_bc()

@pytest.mark.parametrize('x,y', [(2, 3), (-1, 4)])
@ti.test(exclude=ti.cpu, require=ti.extension.extfunc)
def test_asm(x, y):
    @ti.kernel
    def func(x: ti.f32, y: ti.f32) -> ti.f32:
        z = 0.0
        ti.asm('$0 = %0 * %1', inputs=[x, y], outputs=[z])
        return z

    assert func(x, y) == x * y

    sb = ti.SourceBuilder('$0 = %0 * %1', mode='asm')
    @ti.kernel
    def another_func(x: ti.f32, y: ti.f32) -> ti.f32:
        z = 0.0
        sb(inputs=[x, y], outputs=[z])
        return z

    assert another_func(x, y) == x * y
