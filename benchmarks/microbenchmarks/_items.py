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


class DataType(BenchmarkItem):
    name = 'dtype'

    def __init__(self):
        self._items = {
            str(ti.i32): ti.i32,
            str(ti.i64): ti.i64,
            str(ti.f32): ti.f32,
            str(ti.f64): ti.f64
        }


class DataSize(BenchmarkItem):
    name = 'dsize'

    def __init__(self):
        self._items = {}
        for i in range(1, 10):  # [4KB,16KB...256MB]
            size_bytes = (4**i) * 1024  # kibibytes(KiB) = 1024
            self._items[size2tag(size_bytes)] = size_bytes
