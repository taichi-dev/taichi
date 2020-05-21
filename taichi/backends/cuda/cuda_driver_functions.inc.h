// clang-format off


// Driver
PER_CUDA_FUNCTION(init, cuInit, int);
PER_CUDA_FUNCTION(driver_get_version, cuDriverGetVersion, int*);

// Device management
PER_CUDA_FUNCTION(device_get_count, cuDeviceGetCount, int *);
PER_CUDA_FUNCTION(device_get, cuDeviceGet, void *, void *);
PER_CUDA_FUNCTION(device_get_name, cuDeviceGetName, char *, int, void *);
PER_CUDA_FUNCTION(device_get_attribute, cuDeviceGetAttribute, int *, uint32, void *);


// Context management
PER_CUDA_FUNCTION(context_create, cuCtxCreate_v2, void*, int, void *);
PER_CUDA_FUNCTION(context_set_current, cuCtxSetCurrent, void *);
PER_CUDA_FUNCTION(context_get_current, cuCtxGetCurrent, void **);

// Stream management
PER_CUDA_FUNCTION(stream_create, cuStreamCreate, void **, uint32);

// Memory management
PER_CUDA_FUNCTION(memcpy_host_to_device, cuMemcpyHtoD_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy_device_to_host, cuMemcpyDtoH_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(malloc, cuMemAlloc_v2, void *, std::size_t);
PER_CUDA_FUNCTION(malloc_managed, cuMemAllocManaged, void *, std::size_t, uint32);
PER_CUDA_FUNCTION(memset, cuMemsetD8_v2, void *, uint8, std::size_t);
PER_CUDA_FUNCTION(mem_free, cuMemFree_v2, void *);
PER_CUDA_FUNCTION(mem_advise, cuMemAdvise, void *, std::size_t, uint32, uint32);
PER_CUDA_FUNCTION(mem_get_info, cuMemGetInfo_v2, std::size_t *, std::size_t *);

// Module and kernels
PER_CUDA_FUNCTION(module_get_function, cuModuleGetFunction, void **, void *, const char *);
PER_CUDA_FUNCTION(module_load_data_ex, cuModuleLoadDataEx, void **, const char *,
                  uint32, uint32 *, void **)
PER_CUDA_FUNCTION(launch_kernel, cuLaunchKernel, void *, uint32, uint32, uint32,
                  uint32, uint32, uint32, uint32, void *, void **, void **);

// Stream management
PER_CUDA_FUNCTION(stream_synchronize, cuStreamSynchronize, void *);

// Event management
PER_CUDA_FUNCTION(event_create, cuEventCreate, void **, uint32)
PER_CUDA_FUNCTION(event_record, cuEventRecord, void *, uint32)
PER_CUDA_FUNCTION(event_elapsed_time, cuEventElapsedTime, float *, void *, void *);

// clang-format on
