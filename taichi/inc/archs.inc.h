// Potentially supported backends

// CPU archs
PER_ARCH(x64)    // a.k.a. AMD64/x86_64
PER_ARCH(arm64)  // a.k.a. Aarch64, WIP
PER_ARCH(js)     // Javascript, N/A
PER_ARCH(cc)     // C language, WIP

// GPU archs
PER_ARCH(cuda)    // NVIDIA CUDA
PER_ARCH(metal)   // Apple Metal
PER_ARCH(opengl)  // OpenGL Compute Shaders
PER_ARCH(dx)      // Microsoft DirectX, N/A
PER_ARCH(opencl)  // OpenCL, N/A
PER_ARCH(amdgpu)  // AMD GPU, N/A
