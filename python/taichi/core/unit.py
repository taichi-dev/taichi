import taichi


def unit(unit_name):
    def decorator(target_class):
        if hasattr(target_class, '__init__'):
            original_init = target_class.__init__
        else:
            def dummy_init(*args, **kwargs):
                pass
            original_init = dummy_init

        def new_init(self, *args, **kwargs):
            if args:
                impl_name = args[0]
            elif 'name' in kwargs:
                impl_name = kwargs['name']
            else:
                assert False
            self.c = getattr(taichi.core, 'create_' + unit_name)(impl_name)
            self.c.initialize(taichi.misc.util.config_from_dict(kwargs))
            original_init(self, *args, **kwargs)

        target_class.__init__ = new_init

        def new_getattr_(self, key):
            return self.c.__getattribute__(key)

        target_class.__getattr__ = new_getattr_

        return target_class
    return decorator
