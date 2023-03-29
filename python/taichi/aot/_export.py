
from typing import List


class AotExportKernel:
    def __init__(self, f, name: str, ) -> None:
        self.kernel = f
        self.name = name


_aot_kernels: List[AotExportKernel] = []


def export_as(f, *, name: str):
    assert hasattr(f,
                   "_is_wrapped_kernel"), "Only Taichi kernels can be exported"

    record = AotExportKernel(f, name)
    _aot_kernels.append(record)
    return f

def export(f):
    export_as(f, name=f.__name__)
