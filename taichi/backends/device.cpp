#include <taichi/backends/device.h>

namespace taichi {
namespace lang {

DeviceAllocationUnique::~DeviceAllocationUnique() {
    device->dealloc_memory(*this);
}

DevicePtr DeviceAllocation::get_ptr(uint64_t offset) const {
    return DevicePtr{this, offset};
}

void Device::memcpy(DevicePtr dst, DevicePtr src, uint64_t size) {
    // Inter-device copy
    if (dst.allocation->device == src.allocation->device) {
        dst.allocation->device->memcpy_internal(dst, src, size);
    }
    // Intra-device copy
#if TI_WITH_VULKAN && TI_WITH_CUDA

#endif
}

}
}