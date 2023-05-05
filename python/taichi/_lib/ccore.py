import os
import sys
import ctypes

from taichi._lib.utils import package_root


def _load_dll(path):
    try:
        if (
            sys.version_info[0] > 3
            or sys.version_info[0] == 3
            and sys.version_info[1] >= 8
        ):
            dll = ctypes.CDLL(path, winmode=0)
        else:
            dll = ctypes.CDLL(path)
    except OSError:
        return None
    return dll


def load_core_exports_dll():
    bin_path = os.path.join(package_root, "_lib", "core_exports", "bin")
    if os.name == "nt":
        if (
            sys.version_info[0] > 3
            or sys.version_info[0] == 3
            and sys.version_info[1] >= 8
        ):
            os.add_dll_directory(bin_path)
        else:
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        dll_path = os.path.join(bin_path, "taichi_core_exports.dll")
    elif sys.platform == "darwin":
        dll_path = os.path.join(bin_path, "libtaichi_core_exports.dylib")
    else:
        dll_path = os.path.join(bin_path, "taichi_core_exports.so")

    return _load_dll(dll_path)


class TaichiCCore:
    def __init__(self) -> None:
        self._dll = load_core_exports_dll()
        if self._dll is None:
            raise RuntimeError("Cannot load taichi_core_exports.dll")

        # int ticore_hello_world(const char *extra_msg);
        self._dll.ticore_hello_world.argtypes = [ctypes.c_char_p]
        self._dll.ticore_hello_world.restype = ctypes.c_int

    def __getattr__(self, name):
        if name == "hello_world":

            def _hello_world(extra_msg):
                return self._dll.ticore_hello_world(extra_msg.encode("utf-8"))

            return _hello_world
        else:
            raise AttributeError(f"Attribute {name} not found")


taichi_ccore = TaichiCCore()
