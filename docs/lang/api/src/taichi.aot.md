# taichi.aot package

## Submodules

## taichi.aot.module module


### class taichi.aot.module.Module(arch)
Bases: `object`

An AOT module to save and load Taichi kernels.

This module serializes the Taichi kernels for a specific arch. The
serialized module can later be loaded to run on that backend, without the
Python environment.

### Example

Usage:

```
m = ti.aot.Module(ti.metal)
m.add_kernel(foo)
m.add_kernel(bar)

m.save('/path/to/module')

# Now the module file '/path/to/module' contains the Metal kernels
# for running ``foo`` and ``bar``.
```


#### add_field(name, field)
Add a taichi field to the AOT module.


* **Parameters**

    
    * **name** – name of taichi field


    * **field** – taichi field


### Example

Usage:

a = ti.field(ti.f32, shape=(4,4))
b = ti.field(“something”)

m.add_field(a)
m.add_field(b)

# Must add in sequence


#### add_kernel(kernel_fn, name=None)
Add a taichi kernel to the AOT module.


* **Parameters**

    
    * **kernel_fn** (*Function*) – the function decorated by taichi kernel.


    * **name** (*str*) – Name to identify this kernel in the module. If not
    provided, uses the built-in `__name__` attribute of kernel_fn.



#### add_kernel_template(kernel_fn)
Add a taichi kernel (with template parameters) to the AOT module.


* **Parameters**

    **kernel_fn** (*Function*) – the function decorated by taichi kernel.


### Example

Usage:

```
@ti.kernel
def bar_tmpl(a: ti.template()):
  x = a
  # or y = a
  # do something with `x` or `y`

m = ti.aot.Module(arch)
with m.add_kernel_template(bar_tmpl) as kt:
  kt.instantiate(a=x)
  kt.instantiate(a=y)

@ti.kernel
def bar_tmpl_multiple_args(a: ti.template(), b: ti.template())
  x = a
  y = b
  # do something with `x` and `y`

with m.add_kernel_template(bar_tmpl) as kt:
  kt.instantiate(a=x, b=y)
```

## Module contents
