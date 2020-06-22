from .util import *


class TaichiClass:
    is_taichi_class = True

    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def local_var(cls, *entries):
        return cls(entries)

    @classmethod
    @python_scope
    def var(cls, *args, **kwargs):
        entries = cls.get_var_entries(*args, **kwargs)
        return cls(entries)

    @classmethod
    @python_scope
    def get_var_entries(cls, *args, **kwargs):
        raise NotImplementedError

    def is_global(self):
        results = [False for _ in self.entries]
        for i, e in enumerate(self.entries):
            if hasattr(e, 'is_global'):
                if e.is_global():
                    results[i] = True
            assert results[i] == results[0], \
               "Taichi classes with mixed global/local entries are not allowed"
        return results[0]

    @local_scope
    @taichi_scope
    def local_subscript(self, *indices):
        assert len(indices) == 1
        return ret.entries[indices[0]]

    @global_scope
    @taichi_scope
    def global_subscript(self, *indices):
        ret = self.empty_copy()
        for i, e in enumerate(self.entries):
            ret.entries[i] = e.subscript(*indices)
        return ret

    @taichi_scope
    def subscript(self, *indices):
        if self.is_global():
            return self.global_subscript(*indices)
        else:
            return self.local_subscript(*indices)

    @global_scope
    def loop_range(self):
        return self.entries[0].loop_range()

    def get_tensor_members(self):
        ret = []
        for e in self.entries:
            e = e.get_tensor_members()
            ret += e
        return ret

    @taichi_scope
    def variable(self):
        return self.__class__(*(e.variable() for e in self.entries))
