#pragma once

namespace taichi::lang {
namespace metal {

struct MetalDevice;
struct MetalPipeline;
struct MetalResourceBinder;
struct MetalStream;
struct MetalCommandList;

bool is_metal_api_available();

}  // namespace metal
}  // namespace taichi::lang
