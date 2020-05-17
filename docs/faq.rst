Frequently asked questions
==========================

**Q:** Can a user iterate over irregular topology instead of grids, such as tetrahedral meshes, line segment vertices?

**A:** These structures have to be represented using 1D arrays in Taichi. You can still iterate over it using ``for i in x`` or ``for i in range(n)``.
However, at compile time, there's little the Taichi compiler can do for you to optimize it. You can still tweak the data layout to get different runtime cache behaviors and performance numbers.
