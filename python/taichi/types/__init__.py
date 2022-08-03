"""
This module defines data types in Taichi:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- quant: for quantized types, see "https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf"
"""
from taichi.types import quant
from taichi.types.annotations import *
from taichi.types.compound_types import *
from taichi.types.ndarray_type import *
from taichi.types.primitive_types import *
from taichi.types.texture_type import *
from taichi.types.utils import *
