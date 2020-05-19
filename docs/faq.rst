Frequently asked questions
==========================

**Q:** Can a user iterate over irregular topologies (e.g., graphs or tetrahedral meshes) instead of regular grids?

**A:** These structures have to be represented using 1D arrays in Taichi. You can still iterate over them using ``for i in x`` or ``for i in range(n)``.
However, at compile time, there's little the Taichi compiler can do for you to optimize it. You can still tweak the data layout to get different runtime cache behaviors and performance numbers.
