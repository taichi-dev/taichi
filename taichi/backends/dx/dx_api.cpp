#include "taichi/backends/dx/dx_api.h"

namespace taichi {
namespace lang {
namespace directx11 {

bool is_dx_api_available() {
#ifdef TI_WITH_DX11
  return true;
#else
  return false;
#endif
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi