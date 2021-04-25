from taichi.core import ti_core as _ti_core
from taichi.misc.util import config_from_dict


def _unit(unit_name):
    def decorator(target_class):
        if target_class.__init__ != object.__init__:
            original_init = target_class.__init__
        else:

            def dummy_init(*args, **kwargs):
                pass

            original_init = dummy_init

        def new_init(self, name, *args, **kwargs):
            self.c = getattr(_ti_core, 'create_' + unit_name)(name)
            self.c.initialize(config_from_dict(kwargs))
            original_init(self, *args, **kwargs)

        target_class.__init__ = new_init

        def new_getattr_(self, key):
            return self.c.__getattribute__(key)

        target_class.__getattr__ = new_getattr_

        return target_class

    return decorator


@_unit('task')
class Task:
    def run(self, *args):
        return self.c.run(args)
