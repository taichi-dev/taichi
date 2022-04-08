// Don't directly using namespace to avoid conflicting symbols.
namespace tl = ::taichi::lang;

tl::DeviceAllocation *cppcast(Taichi_DeviceAllocation *da) {
  return reinterpret_cast<tl::DeviceAllocation *>(da);
}

tl::Device *cppcast(Taichi_Device *dev) {
  return reinterpret_cast<tl::Device *>(dev);
}
