from microbenchmarks._utils import _size2tag

import taichi as ti


class BenchmarkItem:
    name = 'item'

    def __init__(self):
        self._items = {}  # {tag: xxx, ...}

    def get(self):
        return self._items  #dict

    def get_tags(self):
        return [key for key in self._items]

    def items(self):
        return self._items.items()  #dict.items()

    def tag_in_item(self, tag: str):
        return tag in self._items

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
            self._items[_size2tag(size_bytes)] = size_bytes
