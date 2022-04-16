"""Taichi math module.

The math module supports glsl-style vectors, matrices and functions.
"""
from ._complex import *
from .mathimpl import *
from .vectypes import *

del mathimpl
del vectypes
