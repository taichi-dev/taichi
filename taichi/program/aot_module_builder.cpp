#include "taichi/program/aot_module_builder.h"

#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

void AotModuleBuilder::add(const std::string &identifier, Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend(identifier, kernel);
}

<<<<<<< HEAD
void AotModuleBuilder::add_field(const std::string &identifier,
                                 bool is_scalar,
                                 DataType dt,
                                 std::pair<int, int> shape,
                                 int vector_size) {
  add_per_backend_field(identifier, is_scalar, dt, shape, vector_size);
}

void AotModuleBuilder::add_kernel_template(const std::string &identifier,
                                           const std::string &key,
                                           Kernel *kernel) {
=======
void AotModuleBuilder::add_kernel_template(const std::string &identifier, 
                           const std::string &key, 
                           Kernel *kernel) {
>>>>>>> c596fb80 (dump metal files ok (txt file to fix))
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend_tmpl(identifier, key, kernel);
}

}  // namespace lang
}  // namespace taichi
