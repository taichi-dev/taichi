#pragma once

#include "taichi/backends/device.h"
#include <d3d11.h>

namespace taichi {
namespace lang {
namespace dx {

class DxResourceBinder : public ResourceBinder {
  ~DxResourceBinder() override;
};

class DxPipeline : public Pipeline {
 public:
  DxPipeline(const PipelineSourceDesc &desc, const std::string &name);
  ~DxPipeline() override;
  ResourceBinder *resource_binder() override;
};

}
}  // namespace lang
}  // namespace taichi