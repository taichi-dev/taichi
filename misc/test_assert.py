import taichi as ti
import pytest

'''
ti.init(arch=ti.cuda, debug=True)
ti.set_gdb_trigger(False)

@ti.kernel
def func():
    assert 0

func()
'''

@ti.require(ti.extension.assertion)
@ti.all_archs_with(debug=True)
def test_assert_minimal():
    ti.set_gdb_trigger(False)
    
    @ti.kernel
    def func():
        assert 0

    with pytest.raises(RuntimeError):
        func()
