"""Taichi's AOT (ahead of time) module.

Users can use Taichi as a GPU compute shader/kernel compiler by compiling their
Taichi kernels into an AOT module.
"""
from taichi.aot.module import Module
from taichi.aot.record import *
