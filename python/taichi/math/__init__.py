"""Taichi math module.

The math module supports glsl-style vectors, matrices and functions.
"""
from .mathimpl import *
from .vectypes import *
from .complex import *

del complex
del mathimpl
del vectypes
