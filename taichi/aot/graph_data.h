#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "taichi/aot/module_data.h"

namespace taichi {
namespace lang {
class AotModuleBuilder;
class Ndarray;
namespace aot {
// Currently only scalar and ndarray are supported.
enum ArgKind { SCALAR, NDARRAY, UNKNOWN };

/*
 * Symbolic argument used in building `Dispatch` nodes in the `Graph`.
 */
struct Arg {
  std::string name;
  // TODO: real element dtype = dtype + element_shape
  std::string dtype_name;
  ArgKind tag;
  std::vector<int> element_shape;

  TI_IO_DEF(name, dtype_name, tag, element_shape);
};

/*
 * Runtime value used in graph execution.
 */
struct IValue {
 public:
  uint64 val;
  ArgKind tag;

  static IValue create(const Ndarray &ndarray) {
    return IValue(reinterpret_cast<intptr_t>(&ndarray), ArgKind::NDARRAY);
  }

  template <typename T>
  static IValue create(T v) {
    return IValue(taichi_union_cast_with_different_sizes<uint64>(v),
                  ArgKind::SCALAR);
  }

 private:
  IValue(uint64 val, ArgKind tag) : val(val), tag(tag) {
  }
};
class TI_DLL_EXPORT Kernel {
 public:
  // Rule of 5 to make MSVC happy
  Kernel() = default;
  virtual ~Kernel() = default;
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;
  Kernel(Kernel &&) = default;
  Kernel &operator=(Kernel &&) = default;

  /**
   * @brief Launches the kernel to the device
   *
   * This does not manage the device to host synchronization.
   *
   * @param ctx Host context
   */
  virtual void launch(RuntimeContext *ctx) = 0;

  virtual void save_to_module(AotModuleBuilder *builder) {
    TI_NOT_IMPLEMENTED;
  }
};

struct CompiledDispatch {
  std::string kernel_name;
  std::vector<Arg> symbolic_args;
  Kernel *compiled_kernel{nullptr};

  TI_IO_DEF(kernel_name, symbolic_args);
};

struct CompiledGraph {
  std::vector<CompiledDispatch> dispatches;

  TI_IO_DEF(dispatches);
};

}  // namespace aot
}  // namespace lang
}  // namespace taichi
