"""Taichi math module.

The math module supports glsl-style vectors, matrices and functions.
"""
from ._complex import *
from .mathimpl import *  # pylint: disable=W0622

del mathimpl
