Performance tips
-------------------------------------------

Avoid synchronization: when using GPU, an asynchronous task queue will be maintained. Whenever reading/writing global tensors, a synchronization will be invoked, which leads to idle cycles on CPU/GPU.

Make Use of GPU Shared Memory and L1-d$ ``ti.cache_l1(x)`` will enforce data loads related to ``x`` cached in L1-cache. ``ti.cache_shared(x)`` will allocate shared memory. TODO: add examples
