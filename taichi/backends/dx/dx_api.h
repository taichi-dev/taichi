#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dcompiler.lib")

#include "taichi/common/core.h"
#include <d3d11.h>

TLANG_NAMESPACE_BEGIN

namespace dx {

bool initialize_dx(bool error_tolerance = false);
bool is_dx_api_available();

}

TLANG_NAMESPACE_END