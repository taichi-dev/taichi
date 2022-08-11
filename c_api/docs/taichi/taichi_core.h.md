# Taichi C-API: Core Functionalities

Taichi Core exposes all necessary interfaces to offload AOT modules to Taichi. Here lists the features universally available disregards to any specific backend. The Taichi Core APIs are guaranteed to be forward compatible.

## Definitions

To guarantee a uniform behavior on any platform, we make the following definitions as reference.

```c
//~alias.bool
//~definition.false
//~definition.true
```

A boolean value is represented by an unsigned 32-bit integer. 1 represents a `true` state and 0 represents a `false` state.

```c
//~alias.flags
```

A bit-field of flags is represented by an unsigned 32-bit integer.

```c
//~definition.null_handle
```

A handle is an unsigned 64-bit interger. And a null handle is a handle of zero value.

## Runtime

A runtime is an instance of Taichi targeting an offload devices.

```c
//~handle.runtime
```

A runtime needs to be created with the `enumeration.arch` of the demanded backend device.

```c
//~enumeration.arch
//~function.create_runtime
```

## AOT Module

An AOT module is a pre-compiled collection of compute graphs and kernels.

```c
//~handle.aot_module
```

AOT modules can be loaded from the file system directly.

```c
//~function.load_aot_module
```

## Device Commands

Device commands are interfaces that logical device

## Declarations

`alias.bool`

A boolean value. Can be either `definition.true` or `definition.false`. Assignment with other values could lead to undefined behavior.

`definition.true`

A condition or a predicate is satisfied; a statement is valid.

`definition.false`

A condition or a predicate is not satisfied; a statement is invalid.

`alias.flags`

A bit field that can be used to represent 32 orthogonal flags.

`definition.null_handle`

A sentinal invalid handle that will never be produced from a valid call to Taichi C-API.

`handle.runtime`

Taichi runtime represents an instance of a logical computating device and its internal dynamic states. The user is responsible to synchronize any use of `handle.runtime`.

`handle.aot_module`

An ahead-of-time (AOT) compiled Taichi module, which contains a collection of kernels and compute graphs.

`handle.event`

A synchronization primitive to manage on-device execution flows in multiple queues.

`handle.memory`

A contiguous allocation of on-device memory.

`handle.kernel`

A Taichi kernel that can be launched on device for execution.

`handle.compute_graph`

A collection of Taichi kernels (a compute graph) to be launched on device with predefined order.

`enumeration.arch`

Types of logical offload devices.

`enumeration.data_type`

Elementary (primitive) data types.

`enumeration.argument_type`

Types of kernel and compute graph argument.

`bit_field.memory_usage`

Usages of a memory allocation.

`structure.memory_allocate_info`

Parameters of a newly allocated memory.

`structure.memory_slice`

A subsection of a memory allocation.

`structure.nd_shape`

Multi-dimensional size of an ND-array.

`structure.nd_array`

Multi-dimentional array of dense primitive data.

`union.argument_value`

A scalar or structured argument value.

`structure.argument`

An argument value to feed kernels.

`structure.named_argument`

An named argument value to feed compute graphcs.

`function.create_runtime`

Create a Taichi Runtime with the specified `enumeration.arch`.

`function.destroy_runtime`

Destroy a Taichi Runtime.

`function.allocate_memory`

Allocate a contiguous on-device memory with provided parameters.

`function.free_memory`

Free a memory allocation.

`function.map_memory`

Map an on-device memory to a host-addressible space. The user MUST ensure the device is not being used by any device command before the map.

`function.unmap_memory`

Unmap an on-device memory and make any host-side changes about the memory visible to the device. The user MUST ensure there is no further access to the previously mapped host-addressible space.

`function.create_event`

Create an event primitive.

`function.destroy_event`

Destroy an event primitive.

`function.copy_memory_device_to_device`

Copy the content of a contiguous subsection of on-device memory to another. The two subsections MUST NOT overlap.

`function.launch_kernel`

Launch a Taichi kernel with provided arguments. The arguments MUST have the same count and types in the same order as in the source code.

`function.launch_compute_graph`

Launch a Taichi kernel with provided named arguments. The named arguments MUST have the same count, names and types as in the source code.

`function.signal_event`

Set an event primitive to a signaled state, so the queues waiting upon the event can go on execution. If the event has been signaled before, the event MUST be reset with `function.reset_event`; otherwise it is an undefined behavior.

`function.reset_event`

Set a signaled event primitive back to an unsignaled state.

`function.wait_event`

Wait on an event primitive until it transitions to a signaled state. The user MUST signal the awaited event; otherwise it is an undefined behavior.

`function.submit`

Submit all commands to the logical device for execution. Ensure that any previous device command has been offloaded to the logical computing device.

`function.wait`

Wait until all previously invoked device command has finished execution.

`function.load_aot_module`

Load a precompiled AOT module from the filesystem. `definition.null_handle` is returned if the runtime failed to load the AOT module from the given path.

`function.destroy_aot_module`

Destroy a loaded AOT module and release all related resources.

`function.get_aot_module_kernel`

Get a precompiled Taichi kernel from the AOT module. `definition.null_handle` is returned if the module does not have a kernel of the specified name.

`function.get_aot_module_compute_graph`

Get a precompiled compute graph from the AOt module. `definition.null_handle` is returned if the module does not have a kernel of the specified name.
