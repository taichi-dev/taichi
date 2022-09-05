// clang-format off

// Driver
PER_CUDA_FUNCTION(init, cuInit, int);

// Device management
PER_CUDA_FUNCTION(device_get_count, cuDeviceGetCount, int *);
PER_CUDA_FUNCTION(device_get, cuDeviceGet, void *, void *);
PER_CUDA_FUNCTION(device_get_name, cuDeviceGetName, char *, int, void *);
PER_CUDA_FUNCTION(device_get_attribute, cuDeviceGetAttribute, int *, uint32, void *);


// Context management
PER_CUDA_FUNCTION(context_create, cuCtxCreate_v2, void*, int, void *);
PER_CUDA_FUNCTION(context_set_current, cuCtxSetCurrent, void *);
PER_CUDA_FUNCTION(context_get_current, cuCtxGetCurrent, void **);
PER_CUDA_FUNCTION(context_pop_current, cuCtxPopCurrent, void **);

// Stream management
PER_CUDA_FUNCTION(stream_create, cuStreamCreate, void **, uint32);

// Memory management
PER_CUDA_FUNCTION(memcpy_host_to_device, cuMemcpyHtoD_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy_device_to_host, cuMemcpyDtoH_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy_device_to_device, cuMemcpyDtoD_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy_host_to_device_async, cuMemcpyHtoDAsync_v2, void *, void *, std::size_t, void *);
PER_CUDA_FUNCTION(memcpy_device_to_host_async, cuMemcpyDtoHAsync_v2, void *, void *, std::size_t, void*);
PER_CUDA_FUNCTION(malloc, cuMemAlloc_v2, void **, std::size_t);
PER_CUDA_FUNCTION(malloc_managed, cuMemAllocManaged, void **, std::size_t, uint32);
PER_CUDA_FUNCTION(memset, cuMemsetD8_v2, void *, uint8, std::size_t);
PER_CUDA_FUNCTION(memsetd32, cuMemsetD32_v2, void *, uint32, std::size_t);
PER_CUDA_FUNCTION(mem_free, cuMemFree_v2, void *);
PER_CUDA_FUNCTION(mem_advise, cuMemAdvise, void *, std::size_t, uint32, uint32);
PER_CUDA_FUNCTION(mem_get_info, cuMemGetInfo_v2, std::size_t *, std::size_t *);
PER_CUDA_FUNCTION(mem_get_attribute, cuPointerGetAttribute, void *, uint32, void *);

// Module and kernels
PER_CUDA_FUNCTION(module_get_function, cuModuleGetFunction, void **, void *, const char *);
PER_CUDA_FUNCTION(module_load_data_ex, cuModuleLoadDataEx, void **, const char *,
                  uint32, uint32 *, void **)
PER_CUDA_FUNCTION(launch_kernel, cuLaunchKernel, void *, uint32, uint32, uint32,
                  uint32, uint32, uint32, uint32, void *, void **, void **);
PER_CUDA_FUNCTION(kernel_get_attribute, cuFuncGetAttribute, int *, uint32, void *);
PER_CUDA_FUNCTION(kernel_get_occupancy, cuOccupancyMaxActiveBlocksPerMultiprocessor, int *, void *, int, size_t);

// Stream management
PER_CUDA_FUNCTION(stream_synchronize, cuStreamSynchronize, void *);

// Event management
PER_CUDA_FUNCTION(event_create, cuEventCreate, void **, uint32)
PER_CUDA_FUNCTION(event_destroy, cuEventDestroy, void *)
PER_CUDA_FUNCTION(event_record, cuEventRecord, void *, void *)
PER_CUDA_FUNCTION(event_synchronize, cuEventSynchronize, void *);
PER_CUDA_FUNCTION(event_elapsed_time, cuEventElapsedTime, float *, void *, void *);

// Vulkan interop
PER_CUDA_FUNCTION(import_external_memory, cuImportExternalMemory, CUexternalMemory*, CUDA_EXTERNAL_MEMORY_HANDLE_DESC*)
PER_CUDA_FUNCTION(external_memory_get_mapped_buffer,cuExternalMemoryGetMappedBuffer,CUdeviceptr *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *)
PER_CUDA_FUNCTION(external_memory_get_mapped_mipmapped_array,cuExternalMemoryGetMappedMipmappedArray,CUmipmappedArray *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *)
PER_CUDA_FUNCTION(mipmapped_array_get_level,cuMipmappedArrayGetLevel,CUarray *, CUmipmappedArray, unsigned int)
PER_CUDA_FUNCTION(surf_object_create,cuSurfObjectCreate,CUsurfObject *, const CUDA_RESOURCE_DESC *)
PER_CUDA_FUNCTION(signal_external_semaphore_async,cuSignalExternalSemaphoresAsync,const CUexternalSemaphore * , const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * , unsigned int  , CUstream)
PER_CUDA_FUNCTION(wait_external_semaphore_async,cuWaitExternalSemaphoresAsync,const CUexternalSemaphore * , const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS * , unsigned int  , CUstream)
PER_CUDA_FUNCTION(import_external_semaphore, cuImportExternalSemaphore,CUexternalSemaphore * , const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *)
// clang-format on
