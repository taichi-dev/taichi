# taichi.snode package

## Submodules

## taichi.snode.fields_builder module


### class taichi.snode.fields_builder.FieldsBuilder()
Bases: `object`

A builder that constructs a SNodeTree instance.

Example:

```
x = ti.field(ti.i32)
y = ti.field(ti.f32)
fb = ti.FieldsBuilder()
fb.dense(ti.ij, 8).place(x)
fb.pointer(ti.ij, 8).dense(ti.ij, 4).place(y)

# Afer this line, `x` and `y` are placed. No more fields can be placed
# into `fb`.
#
# The tree looks like the following:
# (implicit root)
#  |
#  +-- dense +-- place(x)
#  |
#  +-- pointer +-- dense +-- place(y)
fb.finalize()
```


#### bit_array(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int], num_bits: int)
Same as `taichi.lang.SNode.bit_array()`


#### bit_struct(num_bits: int)
Same as `taichi.lang.SNode.bit_struct()`


#### bitmasked(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int])
Same as `taichi.lang.SNode.bitmasked()`


#### dense(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int])
Same as `taichi.lang.SNode.dense()`


#### dynamic(index: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimension: Union[Sequence[int], int], chunk_size: Optional[int] = None)
Same as `taichi.lang.SNode.dynamic()`


#### finalize(raise_warning=True)
Constructs the SNodeTree and finalizes this builder.


* **Parameters**

    **raise_warning** (*bool*) – Raise warning or not.



#### classmethod finalized_roots()
Gets all the roots of the finalized SNodeTree.


* **Returns**

    A list of the roots of the finalized SNodeTree.



#### hash(indices, dimensions)
Same as `taichi.lang.SNode.hash()`


#### lazy_grad()
Same as `taichi.lang.SNode.lazy_grad()`


#### place(\*args: Any, offset: Optional[Union[Sequence[int], int]] = None, shared_exponent: bool = False)
Same as `taichi.lang.SNode.place()`


#### pointer(indices: Union[Sequence[taichi_core.Axis], taichi_core.Axis], dimensions: Union[Sequence[int], int])
Same as `taichi.lang.SNode.pointer()`

## Module contents
