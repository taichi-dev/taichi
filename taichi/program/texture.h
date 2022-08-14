#pragma once

#include <cstdint>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/rhi/device.h"

namespace taichi {
namespace lang {

class Program;
class Ndarray;
class SNode;

class TI_DLL_EXPORT Texture {
 public:
  /* Constructs a Texture managed by Program.
   * Texture object allocation and deallocation is handled by Program.
   */
  explicit Texture(Program *prog,
                   BufferFormat format,
                   int width,
                   int height,
                   int depth = 1);

  /* Constructs a Texture from an existing DeviceAllocation
   * It doesn't handle the allocation and deallocation.
   */
  explicit Texture(DeviceAllocation &devalloc,
                   BufferFormat format,
                   int width,
                   int height,
                   int depth = 1);

  intptr_t get_device_allocation_ptr_as_int() const;

  void from_ndarray(Ndarray *ndarray);

  void from_snode(SNode *snode);

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

  Program *prog_{nullptr};
};

}  // namespace lang
}  // namespace taichi
