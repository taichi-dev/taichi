#pragma once
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include "taichi/common/core.h"
#include <d3d11.h>

namespace taichi {
namespace lang {
namespace directx11 {

bool is_dx_api_available();

}
}  // namespace lang
}  // namespace taichi