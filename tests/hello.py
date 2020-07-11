import taichi as ti
import functools
import pytest



@ti.test(ti.cpu, ti.cc, extensions=[ti.extension.sparse], debug=True)
def test_this():
    print('hello')
    print(ti.cfg.arch)
    print(ti.cfg.debug)


@ti.test(ti.cpu)
def test_that():
    print('hello')
    print(ti.cfg.arch)
    print(ti.cfg.debug)
