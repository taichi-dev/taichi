from enum import Enum, unique


@unique
class Layout(Enum):
    """Layout of a Taichi field or ndarray.

    Currently, AOS (array of structures) and SOA (structure of arrays) are supported.
    """
    AOS = 1
    SOA = 2
