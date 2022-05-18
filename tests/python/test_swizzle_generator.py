import pytest
from taichi.lang.swizzle_generator import SwizzleGenerator


def test_swizzle_gen():
    sg = SwizzleGenerator(max_unique_elems=4)
    pats = sg.generate('xyzw', 4)
    uniq_pats = set(pats)
    # https://jojendersie.de/performance-optimal-vector-swizzling-in-c/
    assert len(uniq_pats) == 340
