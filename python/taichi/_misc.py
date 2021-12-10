import sys

from taichi._lib import core as _ti_core
from taichi.lang.enums import Layout  # pylint: disable=unused-import
from taichi.tools import deprecated, warning

from taichi import ad

__all__ = [
    "__version__", "complex_kernel", "complex_kernel_grad", "deprecated_names"
]

deprecated_names = {'SOA': 'Layout.SOA', 'AOS': 'Layout.AOS'}
if sys.version_info.minor < 7:
    for name, alter in deprecated_names.items():
        exec(f'{name} = {alter}')
        __all__.append(name)
else:

    def __getattr__(attr):
        if attr in deprecated_names:
            warning(
                f'ti.{attr} is deprecated. Please use ti.{deprecated_names[attr]} instead.',
                DeprecationWarning,
                stacklevel=2)
            exec(f'{attr} = {deprecated_names[attr]}')
            return locals()[attr]
        raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")

    __all__.append("__getattr__")

complex_kernel = deprecated('ti.complex_kernel',
                            'ti.ad.grad_replaced')(ad.grad_replaced)

complex_kernel_grad = deprecated('ti.complex_kernel_grad',
                                 'ti.ad.grad_for')(ad.grad_for)

__version__ = (_ti_core.get_version_major(), _ti_core.get_version_minor(),
               _ti_core.get_version_patch())
