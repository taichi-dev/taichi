import ctypes

def load_taichi_c_api() -> ctypes.CDLL:
    import ctypes.util as ctypes_util
    from os import environ
    from pathlib import Path

    path = ctypes_util.find_library('taichi_c_api')

    if path is None:
        taichi_c_api_install_dir = environ['TAICHI_C_API_INSTALL_DIR']
        if taichi_c_api_install_dir != None:
            candidate_file_names = [
                'bin/taichi_c_api.dll',
                'lib/libtaichi_c_api.so',
                'lib/libtaichi_c_api.dylib',
            ]
            taichi_c_api_install_dir = Path(taichi_c_api_install_dir)
            for candidate_file_name in candidate_file_names:
                candidate_file_path = taichi_c_api_install_dir / candidate_file_name
                if candidate_file_path.exists():
                    path = str(candidate_file_path)
                    break

    if path is None:
        raise RuntimeError(
            'Cannot find taichi_c_api. Please set TAICHI_C_API_INSTALL_DIR environment variable.'
        )

    print(f'Found taichi_c_api at {path}')
    out = ctypes.CDLL(path, ctypes.RTLD_LOCAL)
    return out

_LIB = load_taichi_c_api()




"""
Handle `TiNsBundle`
"""
TiNsBundle = ctypes.c_void_p


"""
Handle `TiMtlDevice`
"""
TiMtlDevice = ctypes.c_void_p


"""
Handle `TiMtlBuffer`
"""
TiMtlBuffer = ctypes.c_void_p


"""
Handle `TiMtlTexture`
"""
TiMtlTexture = ctypes.c_void_p


"""
Structure `TiMetalRuntimeInteropInfo`
"""
class TiMetalRuntimeInteropInfo(ctypes.Structure): pass
TiMetalRuntimeInteropInfo._fields_ = [
    ('bundle', TiNsBundle),
    ('device', TiMtlDevice),
]


"""
Structure `TiMetalMemoryInteropInfo`
"""
class TiMetalMemoryInteropInfo(ctypes.Structure): pass
TiMetalMemoryInteropInfo._fields_ = [
    ('buffer', TiMtlBuffer),
]


"""
Structure `TiMetalImageInteropInfo`
"""
class TiMetalImageInteropInfo(ctypes.Structure): pass
TiMetalImageInteropInfo._fields_ = [
    ('texture', TiMtlTexture),
]


def ti_import_metal_runtime(
  interop_info: ctypes.c_void_p, # const TiMetalRuntimeInteropInfo*
) -> TiRuntime:
    """
    Function `ti_import_metal_runtime`

    Return value: TiRuntime

    Parameters:
        interop_info (`TiMetalRuntimeInteropInfo`):
    """
    return _LIB.ti_import_metal_runtime(interop_info)


def ti_export_metal_runtime(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # TiMetalRuntimeInteropInfo*
) -> None:
    """
    Function `ti_export_metal_runtime`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiMetalRuntimeInteropInfo`):
    """
    return _LIB.ti_export_metal_runtime(runtime, interop_info)


def ti_import_metal_memory(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # const TiMetalMemoryInteropInfo*
) -> TiMemory:
    """
    Function `ti_import_metal_memory`

    Return value: TiMemory

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiMetalMemoryInteropInfo`):
    """
    return _LIB.ti_import_metal_memory(runtime, interop_info)


def ti_export_metal_memory(
  runtime: TiRuntime,
  memory: TiMemory,
  interop_info: ctypes.c_void_p, # TiMetalMemoryInteropInfo*
) -> None:
    """
    Function `ti_export_metal_memory`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
        interop_info (`TiMetalMemoryInteropInfo`):
    """
    return _LIB.ti_export_metal_memory(runtime, memory, interop_info)


def ti_import_metal_image(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # const TiMetalImageInteropInfo*
) -> TiImage:
    """
    Function `ti_import_metal_image`

    Return value: TiImage

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiMetalImageInteropInfo`):
    """
    return _LIB.ti_import_metal_image(runtime, interop_info)


def ti_export_metal_image(
  runtime: TiRuntime,
  image: TiImage,
  interop_info: ctypes.c_void_p, # TiMetalImageInteropInfo*
) -> None:
    """
    Function `ti_export_metal_image`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
        interop_info (`TiMetalImageInteropInfo`):
    """
    return _LIB.ti_export_metal_image(runtime, image, interop_info)
