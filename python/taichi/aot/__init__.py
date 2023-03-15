"""Taichi's AOT (ahead of time) module.

Users can use Taichi as a GPU compute shader/kernel compiler by compiling their
Taichi kernels into an AOT module.
"""
import taichi.aot.conventions
from taichi.aot.conventions.gfxruntime140 import GfxRuntime140
from taichi.aot.module import Module
from taichi.aot.record import *
import taichi as ti

_aot_kernels = []


def export(f):
    assert hasattr(f,
                   "_is_wrapped_kernel"), "Only Taichi kernels can be exported"
    out = f
    _aot_kernels.append(out)
    return out
