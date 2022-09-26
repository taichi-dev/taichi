---
sidebar_position: 5
---

# SIMT Intrinsics

For the CUDA backend, Taichi supports warp-level and block-level intrinsics, which
are needed for writing high-performance SIMT kernels. You can use them in Taichi
similar to the [usage in CUDA kernels](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/). Currently, the following functions are supported:


| Operation                  | Mapped CUDA intrinsic         |
| -------------------------- | ----------------------------- |
|`ti.simt.warp.all_nonzero`  | `__all_sync`      |
|`ti.simt.warp.any_nonzero`  | `__any_sync`      |
|`ti.simt.warp.unique`       | `__uni_sync`      |
|`ti.simt.warp.ballot`       | `__ballot_sync`   |
|`ti.simt.warp.shfl_sync_i32`| `__shfl_sync`     |
|`ti.simt.warp.shfl_sync_f32`| `__shfl_sync`     |
|`ti.simt.warp.shfl_up_i32`  | `__shfl_up_sync`  |
|`ti.simt.warp.shfl_up_f32`  | `__shfl_up_sync`  |
|`ti.simt.warp.shfl_down_i32`| `__shfl_down_sync`|
|`ti.simt.warp.shfl_down_f32`| `__shfl_down_sync`|
|`ti.simt.warp.shfl_xor_i32` | `__shfl_xor_sync` |
|`ti.simt.warp.match_any`    | `__match_any_sync`|
|`ti.simt.warp.match_all`    | `__match_all_sync`|
|`ti.simt.warp.active_mask`  | `__activemask`    |
|`ti.simt.warp.sync`         | `__syncwarp`      |

See [Taichi's API reference](https://docs.taichi-lang.org/api/taichi/lang/simt/warp/#module-taichi.lang.simt.warp)
for more information on each function.

Here is an example of performing data exchange within a warp in Taichi:


```python
a = ti.field(dtype=ti.i32, shape=32)

@ti.kernel
def foo():
    ti.loop_config(block_dim=32)
    for i in range(32):
        a[i] = ti.simt.warp.shfl_up_i32(ti.u32(0xFFFFFFFF), a[i], 1)

for i in range(32):
    a[i] = i * i

foo()

for i in range(1, 32):
    assert a[i] == (i - 1) * (i - 1)
```
