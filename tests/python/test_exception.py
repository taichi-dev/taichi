from inspect import currentframe, getframeinfo
from sys import version_info

import pytest

import taichi as ti


@ti.test()
def test_exception_multiline():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.TaichiCompilationError) as e:
        # yapf: disable
        @ti.kernel
        def foo():
            aaaa(111,
                 1211222,

                 23)
        foo()
        # yapf: enable

    if version_info < (3, 8):
        msg = f"""\
On line {frameinfo.lineno + 5} of file "{frameinfo.filename}":
            aaaa(111,
TypeError: 'NoneType' object is not callable"""
    else:
        msg = f"""\
On line {frameinfo.lineno + 5} of file "{frameinfo.filename}":
            aaaa(111,
            ^^^^^^^^^
                 1211222,
                 ^^^^^^^^


                 23)
                 ^^^
TypeError: 'NoneType' object is not callable"""
    print(e.value.args[0])
    assert e.value.args[0] == msg


@ti.test()
def test_exception_from_func():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.TaichiCompilationError) as e:

        @ti.func
        def baz():
            t()

        @ti.func
        def bar():
            baz()

        @ti.kernel
        def foo():
            bar()

        foo()
    lineno = frameinfo.lineno
    file = frameinfo.filename
    if version_info < (3, 8):
        msg = f"""\
On line {lineno + 13} of file "{file}":
            bar()
On line {lineno + 9} of file "{file}":
            baz()
On line {lineno + 5} of file "{file}":
            t()
TypeError: 'NoneType' object is not callable"""
    else:
        msg = f"""\
On line {lineno + 13} of file "{file}":
            bar()
            ^^^^^
On line {lineno + 9} of file "{file}":
            baz()
            ^^^^^
On line {lineno + 5} of file "{file}":
            t()
            ^^^
TypeError: 'NoneType' object is not callable"""
    print(e.value.args[0])
    assert e.value.args[0] == msg
