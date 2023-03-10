# GfxRuntime140 convention

GfxRuntime140 is a legacy AOT module convention that serves the Vulkan, Metal and OpenGL backends.

GfxRuntime140 only accepts SPIR-V as the only valid code artifact. For each SPIR-V artifact, all of the following requirements *must* be satisfied.

- SPIR-V module version is 1.3 or higher.
- There is only one entry point function called `main`. It's execution model is `GLCompute`. Workgroup size Y and Z are always `1`.
- There is only one descriptor set, where `Set=0`.
- The context buffer is a uniform bufffer in `std140` layout, bound to `Binding=0`. Its elements are aligned to 4 bytes. The size is at least 1536 bytes. The context buffer content must be consumed following the rules listed below in "Context buffer format".
- The root buffer is a storage buffer in `std430` layout, bound to `Binding=1`.
- ND-arrays are storage buffers in `std430` layout, bound to `Binding>=2` in argument order.

## Context buffer layout

The context buffer has the following constituents sequentially in order:

- Argument values. Size of this section depends on the number of arguments and the types of the arguments.
  - 8-bit scalars take 4 bytes. Valid value in 1 byte from lower address.
  - 16-bit scalars take 4 bytes. Valid value in 2 bytes from lower address.
  - 32-bit scalars take 4 bytes.
  - 64-bit scalars take 8 bytes.
  - N-D arrays and textures take 4 bytes. Never use the values.
- Extra arguments for N-D arrays and textures in the first 32 arguments, 12 extra arguments for each. Exactly take 32 * 12 * 4 = 1536 bytes
  - Scalars zeros all its 12 extra argumnts.
  - N-D arrays stores its shape in the extra arguments. If a N-D array uses less than 12 dimensions, the trailing extra arguments are zeroed.
  - Textures stores its width, height and depth in the extra arguments. The trailing dimensions are zeroed.
