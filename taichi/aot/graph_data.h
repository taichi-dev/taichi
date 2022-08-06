#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "taichi/ir/type.h"
#include "taichi/aot/module_data.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

template <typename T, typename G>
T taichi_union_cast_with_different_sizes(G g);

namespace taichi {
namespace lang {
class AotModuleBuilder;
class Ndarray;
class Texture;

namespace aot {
// Currently only scalar, matrix and ndarray are supported.
enum class ArgKind {
  kScalar,
  kMatrix,
  kNdarray,
  kTexture,
  kRWTexture,
  kUnknown
};

/**
 * Symbolic argument used in building `Dispatch` nodes in the `Graph`.
 */
struct Arg {
  ArgKind tag;
  std::string name;
  // Ndarray: element_dtype = dtype + element_shape
  // Texture: element_shape carries [width, height, depth] info
  //          dtype_id carries channel_format info
  PrimitiveTypeID dtype_id;
  size_t field_dim;
  std::vector<int> element_shape;

  // For texture
  size_t num_channels;  // TODO: maybe rename field_dim and merge?

  // For serialization & deserialization
  explicit Arg()
      : tag(ArgKind::kUnknown),
        name(""),
        dtype_id(PrimitiveTypeID::unknown),
        field_dim(0),
        element_shape({}) {
  }

  explicit Arg(ArgKind tag,
               const std::string &name,

               PrimitiveTypeID dtype_id,
               size_t field_dim,
               const std::vector<int> &element_shape)
      : tag(tag),
        name(name),
        dtype_id(dtype_id),
        field_dim(field_dim),
        element_shape(element_shape) {
  }

  // Python/C++ interface that's user facing.
  explicit Arg(ArgKind tag,
               const std::string &name,
               const DataType &dtype,
               size_t dim = 0,
               const std::vector<int> &element_shape = {})
      : tag(tag), name(name), element_shape(element_shape) {
    if (tag == ArgKind::kTexture || tag == ArgKind::kRWTexture) {
      num_channels = dim;
    } else {
      field_dim = dim;
    }
    dtype_id = dtype->as<PrimitiveType>()->type;
  }

  DataType dtype() const {
    return PrimitiveType::get(dtype_id);
  }

  bool operator==(const Arg &other) const {
    return tag == other.tag && name == other.name &&
           field_dim == other.field_dim && dtype_id == other.dtype_id &&
           element_shape == other.element_shape;
  }

  bool operator!=(const Arg &other) const {
    return !(*this == other);
  }

  TI_IO_DEF(name, dtype_id, field_dim, tag, element_shape, num_channels);
};

/**
 * Runtime value used in graph execution.
 */
struct TI_DLL_EXPORT IValue {
 public:
  uint64 val;
  ArgKind tag;

  static IValue create(const Ndarray &ndarray) {
    return IValue(reinterpret_cast<intptr_t>(&ndarray), ArgKind::kNdarray);
  }

  static IValue create(const Texture &tex) {
    return IValue(reinterpret_cast<intptr_t>(&tex), ArgKind::kTexture);
  }

  template <typename T,
            typename = std::enable_if_t<!std::is_same<T, Ndarray>::value, void>>
  static IValue create(T v) {
    return IValue(taichi_union_cast_with_different_sizes<uint64>(v),
                  ArgKind::kScalar);
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
};

struct CompiledDispatch {
  std::string kernel_name;
  std::vector<Arg> symbolic_args;
  Kernel *compiled_kernel{nullptr};

  TI_IO_DEF(kernel_name, symbolic_args);
};

struct TI_DLL_EXPORT CompiledGraph {
  std::vector<CompiledDispatch> dispatches;
  std::unordered_map<std::string, aot::Arg> args;
  RuntimeContext ctx_;

  void run(const std::unordered_map<std::string, IValue> &args) const;

  TI_IO_DEF(dispatches);
};

}  // namespace aot
}  // namespace lang
}  // namespace taichi
