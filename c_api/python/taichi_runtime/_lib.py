import ctypes

def load_taichi_c_api() -> ctypes.CDLL:
    import ctypes.util as ctypes_util
    from os import environ
    from pathlib import Path

    path = ctypes_util.find_library("taichi_c_api")

    if path is None:
        taichi_c_api_install_dir = environ["TAICHI_C_API_INSTALL_DIR"]
        if taichi_c_api_install_dir != None:
            candidate_file_names = [
                "bin/taichi_c_api.dll",
                "lib/libtaichi_c_api.so",
                "lib/libtaichi_c_api.dylib",
            ]
            taichi_c_api_install_dir = Path(taichi_c_api_install_dir)
            for candidate_file_name in candidate_file_names:
                candidate_file_path = taichi_c_api_install_dir / candidate_file_name
                if candidate_file_path.exists():
                    path = str(candidate_file_path)
                    break

    if path is None:
        raise RuntimeError(
            "Cannot find taichi_c_api. Please set TAICHI_C_API_INSTALL_DIR environment variable."
        )

    print(f"Found taichi_c_api at {path}")
    out = ctypes.CDLL(path, ctypes.RTLD_LOCAL)
    return out


_LIB = load_taichi_c_api()
