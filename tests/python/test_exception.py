from inspect import currentframe, getframeinfo
from sys import version_info

import pytest
from tests import test_utils

import taichi as ti


@test_utils.test()
def test_exception_multiline():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.TaichiNameError) as e:
        # yapf: disable
        @ti.kernel
        def foo():
            aaaa(111,
                 1211222,

                 23)
        foo()
        # yapf: enable

    if version_info < (3, 8):
        msg = f"""
On line {frameinfo.lineno + 5} of file "{frameinfo.filename}", in foo:
            aaaa(111,"""
    else:
        msg = f"""
On line {frameinfo.lineno + 5} of file "{frameinfo.filename}", in foo:
            aaaa(111,
            ^^^^"""
    print(e.value.args[0])
    assert e.value.args[0][:len(msg)] == msg


@test_utils.test()
def test_exception_from_func():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.TaichiNameError) as e:

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
        msg = f"""
On line {lineno + 13} of file "{file}", in foo:
            bar()
On line {lineno + 9} of file "{file}", in bar:
            baz()
On line {lineno + 5} of file "{file}", in baz:
            t()"""
    else:
        msg = f"""
On line {lineno + 13} of file "{file}", in foo:
            bar()
            ^^^^^
On line {lineno + 9} of file "{file}", in bar:
            baz()
            ^^^^^
On line {lineno + 5} of file "{file}", in baz:
            t()
            ^"""
    print(e.value.args[0])
    assert e.value.args[0][:len(msg)] == msg


@test_utils.test()
def test_tab():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.TaichiNameError) as e:
        # yapf: disable
        @ti.kernel
        def foo():
            a(11,	22,	3)
        foo()
        # yapf: enable
    lineno = frameinfo.lineno
    file = frameinfo.filename
    if version_info < (3, 8):
        msg = f"""
On line {lineno + 5} of file "{file}", in foo:
            a(11,   22, 3)"""
    else:
        msg = f"""
On line {lineno + 5} of file "{file}", in foo:
            a(11,   22, 3)
            ^"""
    print(e.value.args[0])
    assert e.value.args[0][:len(msg)] == msg


@test_utils.test()
def test_super_long_line():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.TaichiNameError) as e:
        # yapf: disable
        @ti.kernel
        def foo():
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(111)
        foo()
        # yapf: enable
    lineno = frameinfo.lineno
    file = frameinfo.filename
    if version_info < (3, 8):
        msg = f"""
On line {lineno + 5} of file "{file}", in foo:
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(111)
"""
    else:
        msg = f"""
On line {lineno + 5} of file "{file}", in foo:
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbaaaaaa
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bbbbbbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(111)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"""
    print(e.value.args[0])
    assert e.value.args[0][:len(msg)] == msg


@pytest.mark.skipif(version_info < (3, 8), reason="This is a feature for python>=3.8")
@test_utils.test()
def test_exception_in_node_with_body():
    frameinfo = getframeinfo(currentframe())
    @ti.kernel
    def foo():
        for i in range(1, 2, 3):
            a = 1
            b = 1
            c = 1
            d = 1

    with pytest.raises(ti.TaichiCompilationError) as e:
        foo()
    lineno = frameinfo.lineno
    file = frameinfo.filename
    msg = f"""
On line {lineno + 3} of file "{file}", in foo:
        for i in range(1, 2, 3):
        ^^^^^^^^^^^^^^^^^^^^^^^^
Range should have 1 or 2 arguments, found 3"""
    print(e.value.args[0])
    assert e.value.args[0] == msg

