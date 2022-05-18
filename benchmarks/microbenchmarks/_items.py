from microbenchmarks._utils import size2tag

import taichi as ti


class BenchmarkItem:
    name = 'item'

    def __init__(self):
        self._items = {}  # {'tag': impl, ...}

    def get(self):
        return self._items

    def get_tags(self):
        return list(self._items.keys())

    def impl(self, tag: str):
        return self._items[tag]

    def remove(self, tags: list):
        for tag in tags:
            self._items.pop(tag)

    def update(self, adict: dict):
        self._items.update(adict)


class DataType(BenchmarkItem):
    name = 'dtype'
    integer_list = ['i32', 'i64']

    def __init__(self):
        self._items = {
            str(ti.i32): ti.i32,
            str(ti.i64): ti.i64,
            str(ti.f32): ti.f32,
            str(ti.f64): ti.f64
        }

    def remove_integer(self):
        self.remove(self.integer_list)

    @staticmethod
    def is_integer(dtype: str):
        integer_list = ['i32', 'u32', 'i64', 'u64']
        return True if dtype in integer_list else False


class DataSize(BenchmarkItem):
    name = 'dsize'

    def __init__(self):
        self._items = {}
        for i in range(2, 10, 2):  # [16KB,256KB,4MB,64MB]
            size_bytes = (4**i) * 1024  # kibibytes(KiB) = 1024
            self._items[size2tag(size_bytes)] = size_bytes


class Container(BenchmarkItem):
    name = 'container'

    def __init__(self):
        self._items = {'field': ti.field, 'ndarray': ti.ndarray}


class MathOps(BenchmarkItem):
    name = 'math_op'

    #reference: https://docs.taichi-lang.org/docs/operator
    def __init__(self):
        self._items = {
            # Trigonometric
            'sin': ti.sin,
            'cos': ti.cos,
            'tan': ti.tan,
            'asin': ti.asin,
            'acos': ti.acos,
            'tanh': ti.tanh,
            # Other arithmetic
            'sqrt': ti.sqrt,
            'rsqrt': ti.rsqrt,  # A fast version for `1 / ti.sqrt(x)`.
            'exp': ti.exp,
            'log': ti.log,
            'round': ti.round,
            'floor': ti.floor,
            'ceil': ti.ceil,
            'abs': ti.abs,
        }


class AtomicOps(BenchmarkItem):
    name = 'atomic_op'

    def __init__(self):
        self._items = {
            'atomic_add': ti.atomic_add,
            'atomic_sub': ti.atomic_sub,
            'atomic_and': ti.atomic_and,
            'atomic_or': ti.atomic_or,
            'atomic_xor': ti.atomic_xor,
            'atomic_max': ti.atomic_max,
            'atomic_min': ti.atomic_min
        }

    @staticmethod
    def is_logical_op(op: str):
        logical_op_list = ['atomic_and', 'atomic_or', 'atomic_xor']
        return True if op in logical_op_list else False

    @staticmethod
    def is_supported_type(op: str, dtype: str):
        if AtomicOps.is_logical_op(op) and not DataType.is_integer(dtype):
            return False
        else:
            return True
