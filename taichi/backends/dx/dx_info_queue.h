#pragma once

#include "taichi/backends/device.h"
#include <d3d11.h>

namespace taichi {
namespace lang {
namespace directx11 {

class Dx11InfoQueue {
 public:
  struct Entry {
    std::string type;
    void *addr;
    int refcount;
    int intref;
  };
  static std::vector<Entry> parse_reference_count(
      const std::vector<std::string>&);
  explicit Dx11InfoQueue(ID3D11Device *device);
  int live_object_count();

 private:
  bool has_updated_messages();
  std::vector<std::string> get_updated_messages();
  std::vector<Entry> live_objects_;
  void init();
  ID3D11Device *device_{};
  ID3D11Debug *debug_{};
  ID3D11InfoQueue *info_queue_{};
  int last_message_count_;
};

}  // namespace directx11
}  // namespace lang
}  // namespace taichi
