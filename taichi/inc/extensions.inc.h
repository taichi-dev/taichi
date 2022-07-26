// Lists of extension features
PER_EXTENSION(sparse)       // Sparse data structures
PER_EXTENSION(quant)        // Quantization
PER_EXTENSION(mesh)         // MeshTaichi
PER_EXTENSION(quant_basic)  // Basic operations in quantization
PER_EXTENSION(data64)       // Metal doesn't support 64-bit data buffers yet...
PER_EXTENSION(adstack)    // For keeping the history of mutable local variables
PER_EXTENSION(bls)        // Block-local storage
PER_EXTENSION(assertion)  // Run-time asserts in Taichi kernels
PER_EXTENSION(extfunc)    // Invoke external functions or backend source
PER_EXTENSION(packed)     // Shape will not be padded to a power of two
PER_EXTENSION(
    dynamic_index)  // Dynamic index support for both global and local tensors
