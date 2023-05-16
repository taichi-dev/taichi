from typing import Any, Dict, List, Optional


class AotExportKernel:
    def __init__(self, f, name: str, template_types: Dict[str, Any]) -> None:
        self.kernel = f
        self.name = name
        self.template_types = template_types


_aot_kernels: List[AotExportKernel] = []


def export_as(name: str, *, template_types: Optional[Dict[str, Any]] = None):
    def inner(f):
        assert hasattr(f, "_is_wrapped_kernel"), "Only Taichi kernels can be exported"

        record = AotExportKernel(f, name, template_types or {})
        _aot_kernels.append(record)
        return f

    return inner


def export(f):
    return export_as(f.__name__)(f)
