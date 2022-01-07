# [RFC] AOT and prod deployment for sparse / quant / mesh

## Goals

- Make Taichi deployable with full functionalities (including sparse / quant / mesh)
- Users can manage the ownership of the system resources (including RAM, VRAM, GPU devices, command lists)
- The provided Taichi runtime API should be independent of arch as much as possible

## Current Problems

Here is a code snippet in Taichi v0.8.8:

```python
a = ti.field(ti.i32)
b = ti.field(ti.f32)
ti.root.pointer(ti.ij, 16).dense(ti.ij, 16).place(a, b)

@ti.kernel
def run():
  for I in ti.grouped(a):
    b[I] = a[I] * 4.2
```

It is hard to deploy this code with AOT: `a` and `b` are runtime resources, which will couple kernel invocation of `run` and memory allocation for `a` and `b`.

It is not a good idea to pack all these things into a module. The memory allocation process must be deployed to user applications, which will violate Goal 2.

Notice that a SNodeTree is in fact a type. To solve this problem, we need to avoid memory allocation for `a` and `b`, but only keep their types (which is a SNodeTree). We can bind the actual physical memory after compilation.

## Proposal

### Python API

Taichi AOT modules should only contain static program and data, like this:

```python
# This program is only for demonstration, it can't run on current version
vec3 = ti.types.Vector(ti.f32, 3)
s = ti.types.Struct({'a': ti.f32, 'b': vec3})

# AOS
sn1 = ti.types.SNodeTreeBuilder()
                .pointer(ti.ij, 3)
                .dense(ti.ij, 5)
                .place(s)  # places s.a, s.b in an AOS way
                .build()

# SOA
sn2b = ti.types.SNodeTreeBuilder()
sn2b
  .pointer(ti.ij, 3)
  .dense(ti.ij, 5)
  .place(s.a)           
sn2b
  .pointer(ti.ij, 3)
  .dense(ti.ij, 5)
  .place(s.b)
sn2 = sn2b.build()

@ti.kernel
def run1(f: sn1):
  for I in ti.grouped(f):
    v = f[I].a * f[I].b
    print('I=', I, ' a*b=', v)
```

To save the compiled results:

```python
m = ti.aot.ModuleBilder(arch=ti.cuda)

m.add_type(sn1, name='ab_aos')
m.add_type(sn2, name='ab_soa')
m.add_kernel(run1)

m.run_optimizations(level=2)
m.save(output_dir="$HOME")
```

### C++ API

It is similar to #3642.

```c++
taichi::Module m = taichi::LoadModule("$HOME/aot_module");
m.arch();  // "cuda"

auto *ab_aos_t = m.GetSNodeTreeType("ab_aos");
const size_t tree_size = ab_aos_t.GetSize();
// Allocate CUDA memory/ VkBuffer using |tree_size} however you want
auto mem = AllocateDeviceMemory(tree_size);
auto typed_mem = taichi::MakeMemoryResourceTyped(mem, ab_aos_t);

auto *run1 = m.GetKernel("run1");
run1.signature();  // {args: {ab_aos_t,}, returns: {}}
run1.Run(/*args=*/{typed_mem});
```

Except `GetDeviceMemory` depends on arch (like `void *` for CUDA, `VkBuffer` for Vulkan), other parts will be independent of arch.

## TBD

- Should the shape of SNodeTree be static?
- Should the compiled results support multiple arch?
- Is it possible to support compute graph for host logic?
- How to support non-field structures (like `Mesh`)?