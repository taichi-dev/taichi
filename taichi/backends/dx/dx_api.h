#pragma once
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

#include "taichi/common/core.h"

#ifdef TI_WITH_DX11
#include <d3d11.h>
#endif

namespace taichi {
namespace lang {
namespace directx11 {

bool is_dx_api_available();

}
}  // namespace lang
}  // namespace taichi
