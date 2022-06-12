#pragma once

#include <cstdint>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/backends/device.h"

namespace taichi {
namespace lang {

class Program;
class Ndarray;

class TI_DLL_EXPORT Texture {
 public:
  /* Constructs a Ndarray managed by Program.
   * Memory allocation and deallocation is handled by Program.
   * TODO: Ideally Ndarray shouldn't worry about memory alloc/dealloc at all.
   */
  explicit Texture(Program *prog,
                   const DataType type,
                   int num_channels,
                   int width,
                   int height,
                   int depth = 1);

  /* Constructs a Ndarray from an existing DeviceAllocation
   * It doesn't handle the allocation and deallocation.
   */
  explicit Texture(DeviceAllocation &devalloc,
                   const DataType type,
                   int num_channels,
                   int width,
                   int height,
                   int depth = 1);

  intptr_t get_device_allocation_ptr_as_int() const;

  void from_ndarray(Ndarray *ndarray);

  DeviceAllocation get_device_allocation() const {
    return texture_alloc_;
  }

  ~Texture();

 private:
  DeviceAllocation texture_alloc_{kDeviceNullAllocation};
  DataType dtype_;
  BufferFormat format_;
  int num_channels_{4};
  int width_;
  int height_;
  int depth_;

  BufferFormat get_format(DataType type, int num_channels);

  Program *prog_{nullptr};
};

}  // namespace lang
}  // namespace taichi
