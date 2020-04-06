// clang-format off
PER_CUDA_FUNCTION(memcpy_host_to_device, cuMemcpyHtoD_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpy_device_to_host, cuMemcpyDtoH_v2, void *, void *, std::size_t);
// clang-format on
