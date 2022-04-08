// Don't directly using namespace to avoid conflicting symbols.
namespace tl = ::taichi::lang;

tl::aot::Module *cppcast(Taichi_AotModule *m) {
  return reinterpret_cast<tl::aot::Module *>(m);
}
