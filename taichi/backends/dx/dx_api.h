#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3dcompiler.lib")
#pragma comment (lib, "dxguid.lib")

#include "taichi/common/core.h"
#include "taichi/backends/device.h"
#include <d3d11.h>
#include <d3dcompiler.h>

TLANG_NAMESPACE_BEGIN

namespace directx11 {

bool initialize_dx(bool error_tolerance = false);
bool is_dx_api_available();
std::unique_ptr<Device> get_dx_device();

ID3D11Device *GetD3D11Device();
ID3D11DeviceContext *GetD3D11Context();
ID3D11Buffer *GetTmpArgBuf();
HRESULT CreateRawBuffer(ID3D11Device *device,
                        UINT size,
                        void *init_data,
                        ID3D11Buffer **out_buf);
HRESULT CreateStructuredBuffer(ID3D11Device *device,
                               UINT element_size,
                               UINT count,
                               void *init_data,
                               ID3D11Buffer **out_buf);
HRESULT CreateCPUAccessibleCopyOfBuffer(ID3D11Device *device,
                                        ID3D11Buffer *src_buf,
                                        ID3D11Buffer **out_buf);
HRESULT CompileComputeShaderFromString(const std::string &source,
                                       LPCSTR entry_point,
                                       ID3D11Device *device,
                                       ID3DBlob **blob);
HRESULT CreateBufferUAV(ID3D11Device *device,
                        ID3D11Buffer *buffer,
                        ID3D11UnorderedAccessView **out_uav);

}

TLANG_NAMESPACE_END