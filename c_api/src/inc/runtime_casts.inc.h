// Don't directly using namespace to avoid conflicting symbols.
namespace tl = ::taichi::lang;

tl::aot::Module *cppcast(AotModule *m) {
  return reinterpret_cast<tl::aot::Module *>(m);
}

tl::DeviceAllocation *cppcast(DeviceAllocation *da) {
  return reinterpret_cast<tl::DeviceAllocation *>(da);
}
