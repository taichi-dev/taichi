#pragma once

#include <unordered_map>

#include "taichi/program/kernel.h"
#include "taichi/program/ndarray.h"

namespace taichi {
namespace lang {

class Program;
class Ndarray;

/* Note: [Ndarray host reader & writer]
 * Unlike snodes/fields which are persistent global storage that can safely
 * use SNode* as keys to cache reader & writer kernels, ndarrays' life-cycle
 * depends on their corresponding python objects. In other words we cannot
 * use Ndarray* here as caching keys since it's possible that one ndarray reuses
 * exactly the same address where a freed ndarray instance was.
 *
 * Fortunately since ndarray reader & writer don't hardcode ndarray address in
 * the kernel, their caching mechanism can also be more efficient than the snode
 * ones. Currently we only use ndarray's num_active_indices & dtype information
 * (saved in NdarrayRwKeys) in the reader & writer kernels Details can be found
 * in get_ndarray_reader/writer in program.cpp.
 */
struct NdarrayRwKeys {
  size_t num_active_indices;
  DataType dtype;

  struct Hasher {
    std::size_t operator()(const NdarrayRwKeys &k) const {
      auto h1 = std::hash<int>{}(k.num_active_indices);
      auto h2 = k.dtype.hash();
      return h1 ^ h2;
    }
  };

  bool operator==(const NdarrayRwKeys &other) const {
    return num_active_indices == other.num_active_indices &&
           dtype == other.dtype;
  }
};

/** A mapping from a Ndarray to its read/write access kernels.
 */
class NdarrayRwAccessorsBank {
 private:
  struct RwKernels {
    Kernel *reader{nullptr};
    Kernel *writer{nullptr};
  };

 public:
  class Accessors {
   public:
    explicit Accessors(const Ndarray *ndarray,
                       const RwKernels &kernels,
                       Program *prog);

    // for float and double
    void write_float(const std::vector<int> &I, float64 val);
    float64 read_float(const std::vector<int> &I);

    // for int32 and int64
    void write_int(const std::vector<int> &I, int64 val);
    int64 read_int(const std::vector<int> &I);
    uint64 read_uint(const std::vector<int> &I);

   private:
    const Ndarray *ndarray_;
    Program *prog_;
    Kernel *reader_;
    Kernel *writer_;
  };

  explicit NdarrayRwAccessorsBank(Program *program) : program_(program) {
  }

  Accessors get(Ndarray *ndarray);

 private:
  Program *const program_;
  std::unordered_map<NdarrayRwKeys, RwKernels, NdarrayRwKeys::Hasher>
      ndarray_to_kernels_;
};

}  // namespace lang
}  // namespace taichi
