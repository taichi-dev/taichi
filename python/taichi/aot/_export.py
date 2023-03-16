import taichi as ti

_aot_kernels = []


def export(f):
    assert hasattr(f,
                   "_is_wrapped_kernel"), "Only Taichi kernels can be exported"
    out = f
    _aot_kernels.append(out)
    return out
