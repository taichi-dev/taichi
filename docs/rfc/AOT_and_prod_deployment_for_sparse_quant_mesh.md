# [RFC] AOT and prod deployment for sparse / quant / mesh

## Goals

1. Make Taichi deployable with full functionalities (including sparse / quant / mesh)
2. Users can manage the ownership of the system resources (including RAM, VRAM, GPU devices, command lists)
3. The provided Taichi runtime API should be independent of arch as much as possible

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

It is hard to deploy the above code snippet using AOT, because the static program and the runtime resources are coupled. the run kernel is the static program code, while a and b represent runtime memory resources. Assuming we would like to package them into an AOT module, there are two options here:

1. The logic for allocating a and b is included inside the module. Upon the module loading time, they will be allocated as well. This is what we want to prevent (violating Goal 2).
2. The logic is not included. However, we still need to include their SNode type information (i.e., the SNodeTree produced by their definition). This complicates the AOT solution.

Notice that a SNodeTree is in fact a type. To solve this problem, we need to avoid memory allocation for `a` and `b`, but only keep their types (which is a SNodeTree). We can bind the actual physical memory after compilation.

## Proposal

### Python API

Taichi AOT modules should only contain the static program data, including functions, kernels and types. Here's one possible API:

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
- Shall we allow users to place multiple structs inside a single SNodeTree builder?
- How to support non-field structures (like `Mesh`)?
