#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

#include "taichi/common/core.h"
#include "taichi/backends/device.h"
#include <d3d11.h>
#include <d3dcompiler.h>

namespace taichi {
namespace lang {
namespace directx11 {

bool is_dx_api_available();

HRESULT create_raw_buffer(ID3D11Device *device,
                          UINT size,
                          void *init_data,
                          ID3D11Buffer **out_buf);
HRESULT create_structured_buffer(ID3D11Device *device,
                                 UINT element_size,
                                 UINT count,
                                 void *init_data,
                                 ID3D11Buffer **out_buf);
HRESULT create_cpu_accessible_buffer_copy(ID3D11Device *device,
                                          ID3D11Buffer *src_buf,
                                          ID3D11Buffer **out_buf);
HRESULT compile_compute_shader_from_string(const std::string &source,
                                           LPCSTR entry_point,
                                           ID3D11Device *device,
                                           ID3DBlob **blob);
HRESULT create_buffer_uav(ID3D11Device *device,
                          ID3D11Buffer *buffer,
                          ID3D11UnorderedAccessView **out_uav);

}  // namespace directx11
}  // namespace lang
}  // namespace taichi