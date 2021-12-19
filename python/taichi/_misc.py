import sys

from taichi._lib import core as _ti_core
from taichi.lang.enums import Layout  # pylint: disable=unused-import
from taichi.tools import warning

__all__ = ["__version__"]

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

__version__ = (_ti_core.get_version_major(), _ti_core.get_version_minor(),
               _ti_core.get_version_patch())
