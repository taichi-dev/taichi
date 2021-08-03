#pragma once

#include <unordered_map>

#include "taichi/program/kernel.h"
#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {

class Program;

/** A mapping from an SNode to its read/write access kernels.
 *
 * The main purpose of this class is to decouple the accessor kernels from the
 * SNode class itself. Ideally, SNode should be nothing more than a group of
 * plain data.
 */
class SNodeRwAccessorsBank {
 private:
  struct RwKernels {
    Kernel *reader{nullptr};
    Kernel *writer{nullptr};
  };

 public:
  class Accessors {
   public:
    explicit Accessors(const SNode *snode,
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
    const SNode *snode_;
    Program *prog_;
    Kernel *reader_;
    Kernel *writer_;
  };

  explicit SNodeRwAccessorsBank(Program *program) : program_(program) {
  }

  Accessors get(SNode *snode);

 private:
  Program *const program_;
  std::unordered_map<const SNode *, RwKernels> snode_to_kernels_;
};

}  // namespace lang
}  // namespace taichi
