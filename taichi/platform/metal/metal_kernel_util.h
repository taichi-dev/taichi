#pragma once

#include <string>
#include <vector>

#include "taichi/ir/statements.h"
#include "taichi/platform/metal/metal_data_types.h"
#include "taichi/program/kernel.h"

// Data structures defined in this file may overlap with some of the Taichi data
// structures. However, they serve as a boundary between Taichi and Metal and
// help decouple the system.
//
// Please keep in mind that each Taichi kernel defined by @ti.kernel can be
// compiled to more than one Metal compute kernels. Concretely, each offloaded
// task in the Taichi kernel maps to a Metal kernel.

TLANG_NAMESPACE_BEGIN

class SNode;

namespace metal {

struct StructCompiledResult {
  // Source code of the SNode data structures compiled to Metal
  std::string source_code;
  // Root buffer size in bytes.
  size_t root_size;
};

// This struct holds the necesary information to launch a Metal kernel.
struct MetalKernelAttributes {
  std::string name;
  int num_threads;
  OffloadedStmt::TaskType task_type;

  struct RangeForAttributes {
    // |begin| has differen meanings depending on |const_begin|:
    // * true : It is the left boundary of the loop known at compile time.
    // * false: It is the offset of the begin in the global tmps buffer.
    //
    // Same applies to |end|.
    size_t begin;
    size_t end;
    bool const_begin{true};
    bool const_end{true};

    inline bool const_range() const {
      return (const_begin && const_end);
    }
  };
  // Only valid when |task_type| is range_for.
  // TODO(k-ye): Use std::optional to wrap this.
  RangeForAttributes range_for_attribs;
};

// Note that all Metal kernels belonging to the same Taichi kernel will share
// the same kernel args (attributes + Metal buffer). This is because kernel
// arguments is a Taichi-level concept.
class MetalKernelArgsAttributes {
 public:
  // Attribute for a single argument.
  // This is mostly the same as Kernel::Arg. It's extended to contain a few
  // Metal kernel specific atrributes, like |stride| and |offset_in_mem|.
  struct ArgAttributes {
    // For array arg, this is #elements * stride(dt). Unit: byte
    size_t stride = 0;
    // Offset in the argument buffer
    size_t offset_in_mem = 0;
    // Argument index
    int index = -1;
    MetalDataType dt;
    bool is_array = false;
    bool is_return_val = false;
  };

  explicit MetalKernelArgsAttributes(const std::vector<Kernel::Arg>& args);

  inline bool has_args() const { return !arg_attribs_vec_.empty(); }
  inline const std::vector<ArgAttributes> &args() const { return arg_attribs_vec_; }

  inline size_t args_bytes() const {
    return args_bytes_;
  }
  inline size_t extra_args_bytes() const {
    return extra_args_bytes_;
  }
  inline size_t total_bytes() const {
    return args_bytes_ + extra_args_bytes_;
  }
 private:
  std::vector<ArgAttributes> arg_attribs_vec_;
  size_t args_bytes_;
  size_t extra_args_bytes_;
};

}  // namespace metal

TLANG_NAMESPACE_END
