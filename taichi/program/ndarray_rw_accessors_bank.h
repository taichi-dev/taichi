#pragma once

#include <unordered_map>

#include "taichi/program/kernel.h"
#include "taichi/program/ndarray.h"

namespace taichi {
namespace lang {

class Program;
class Ndarray;

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
  std::unordered_map<const Ndarray *, RwKernels> ndarray_to_kernels_;
};

}  // namespace lang
}  // namespace taichi
