#include <functional>

#include "pybind11/pybind11.h"
#include "taichi/common/interface.h"
#include "taichi/common/task.h"
#include "taichi/system/benchmark.h"

namespace taichi {

#define TI_INTERFACE_DEF_WITH_PYBIND11(class_name, base_alias)                \
                                                                              \
  class InterfaceInjector_##class_name {                                      \
   public:                                                                    \
    InterfaceInjector_##class_name(const std::string &name) {                 \
      InterfaceHolder::get_instance()->register_registration_method(          \
          base_alias, [&](void *m) {                                          \
            ((pybind11::module *)m)                                           \
                ->def("create_" base_alias,                                   \
                      static_cast<std::shared_ptr<class_name> (*)(            \
                          const std::string &name)>(                          \
                          &create_instance<class_name>));                     \
            ((pybind11::module *)m)                                           \
                ->def("create_initialized_" base_alias,                       \
                      static_cast<std::shared_ptr<class_name> (*)(            \
                          const std::string &name, const Config &config)>(    \
                          &create_instance<class_name>));                     \
          });                                                                 \
      InterfaceHolder::get_instance()->register_interface(                    \
          base_alias, (ImplementationHolderBase *)                            \
                          get_implementation_holder_instance_##class_name()); \
    }                                                                         \
  } ImplementationInjector_##base_class_name##class_name##instance(base_alias);

TI_INTERFACE_DEF_WITH_PYBIND11(Benchmark, "benchmark")
TI_INTERFACE_DEF_WITH_PYBIND11(Task, "task")

}  // namespace taichi
