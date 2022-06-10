import os

import taichi as ti


def compile_aot():
    # Make sure "debug" mode is on
    # in both python & C++ tests
    ti.init(arch=ti.x64, debug=True)

    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    common = ti.root.dense(ti.i, 4)
    common.dense(ti.i, 8).place(x)

    p = common.pointer(ti.i, 2)
    p.dense(ti.i, 8).place(y)

    @ti.kernel
    def init_fields(base: int):
        # Dense SNode
        for i in range(4 * 8):
            x[i] = base + i

        # Pointer SNode
        y[32] = 4
        y[33] = 5
        y[9] = 10

    @ti.kernel
    def check_init_x(base: int):
        # Check numerical accuracy for Dense SNodes
        for i in range(4 * 8):
            assert (x[i] == base + i)

    @ti.kernel
    def check_init_y():
        # Check sparsity for Pointer SNodes
        for i in range(8):
            if i == 1 or i == 4:
                assert (ti.is_active(p, [i]))
            else:
                assert (not ti.is_active(p, [i]))

        # Check numerical accuracy for Pointer SNodes
        for i in range(8, 8 + 8):
            if i == 9:
                assert (y[i] == 10)
            else:
                assert (y[i] == 0)

        for i in range(32, 32 + 8):
            if i == 32:
                assert (y[i] == 4)
            elif i == 33:
                assert (y[i] == 5)
            else:
                assert (y[i] == 0)

    @ti.kernel
    def deactivate_pointer_fields():
        ti.deactivate(p, [1])
        ti.deactivate(p, [4])

    @ti.kernel
    def activate_pointer_fields():
        ti.activate(p, [7])
        ti.activate(p, [3])

    @ti.kernel
    def check_deactivate_pointer_fields():
        assert (not ti.is_active(p, [1]))
        assert (not ti.is_active(p, [4]))

    @ti.kernel
    def check_activate_pointer_fields():
        assert (ti.is_active(p, [7]))
        assert (ti.is_active(p, [3]))

        for i in range(7 * 8, 7 * 8 + 8):
            assert (y[i] == 0)

        for i in range(3 * 8, 3 * 8 + 8):
            assert (y[i] == 0)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module(ti.x64)

    m.add_kernel(init_fields, template_args={})
    m.add_kernel(check_init_x, template_args={})
    m.add_kernel(check_init_y, template_args={})

    m.add_kernel(deactivate_pointer_fields, template_args={})
    m.add_kernel(activate_pointer_fields, template_args={})

    m.add_kernel(check_deactivate_pointer_fields, template_args={})
    m.add_kernel(check_activate_pointer_fields, template_args={})

    m.add_field("x", x)
    m.add_field("y", y)

    m.save(dir_name, 'x64-aot')


compile_aot()
