import taichi as ti


def export(kern):
    kern = kern._primal
    assert ti.cfg.arch == ti.opengl  # for now (@yuanming-hu & @k-ye)
    # state machine hack to make taichi KernelProxy happy
    ti.core.set_export_flag(True)
    kern.materialize()  # what about key & args & arg_features?
    ti.core.set_export_flag(False)
    src = ti.core.get_last_exported_source()
    return src
