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
Structure `TiOpenglRuntimeInteropInfo`
"""
class TiOpenglRuntimeInteropInfo(ctypes.Structure): pass
TiOpenglRuntimeInteropInfo._fields_ = [
    ('get_proc_addr', ctypes.c_void_p),
]


"""
Structure `TiOpenglMemoryInteropInfo`
"""
class TiOpenglMemoryInteropInfo(ctypes.Structure): pass
TiOpenglMemoryInteropInfo._fields_ = [
    ('buffer', GLuint),
    ('size', GLsizeiptr),
]


"""
Structure `TiOpenglImageInteropInfo`
"""
class TiOpenglImageInteropInfo(ctypes.Structure): pass
TiOpenglImageInteropInfo._fields_ = [
    ('texture', GLuint),
    ('target', GLenum),
    ('levels', GLsizei),
    ('format', GLenum),
    ('width', GLsizei),
    ('height', GLsizei),
    ('depth', GLsizei),
]


def ti_import_opengl_runtime(
  interop_info: ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*
) -> TiRuntime:
    """
    Function `ti_import_opengl_runtime`

    Return value: TiRuntime

    Parameters:
        interop_info (`TiOpenglRuntimeInteropInfo`):
    """
    return _LIB.ti_import_opengl_runtime(interop_info)


def ti_export_opengl_runtime(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*
) -> None:
    """
    Function `ti_export_opengl_runtime`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiOpenglRuntimeInteropInfo`):
    """
    return _LIB.ti_export_opengl_runtime(runtime, interop_info)


def ti_import_opengl_memory(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # const TiOpenglMemoryInteropInfo*
) -> TiMemory:
    """
    Function `ti_import_opengl_memory`

    Return value: TiMemory

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiOpenglMemoryInteropInfo`):
    """
    return _LIB.ti_import_opengl_memory(runtime, interop_info)


def ti_export_opengl_memory(
  runtime: TiRuntime,
  memory: TiMemory,
  interop_info: ctypes.c_void_p, # TiOpenglMemoryInteropInfo*
) -> None:
    """
    Function `ti_export_opengl_memory`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
        interop_info (`TiOpenglMemoryInteropInfo`):
    """
    return _LIB.ti_export_opengl_memory(runtime, memory, interop_info)


def ti_import_opengl_image(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # const TiOpenglImageInteropInfo*
) -> TiImage:
    """
    Function `ti_import_opengl_image`

    Return value: TiImage

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiOpenglImageInteropInfo`):
    """
    return _LIB.ti_import_opengl_image(runtime, interop_info)


def ti_export_opengl_image(
  runtime: TiRuntime,
  image: TiImage,
  interop_info: ctypes.c_void_p, # TiOpenglImageInteropInfo*
) -> None:
    """
    Function `ti_export_opengl_image`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
        interop_info (`TiOpenglImageInteropInfo`):
    """
    return _LIB.ti_export_opengl_image(runtime, image, interop_info)
