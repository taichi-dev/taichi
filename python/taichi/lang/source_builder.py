import atexit
import ctypes
import os
import shutil
import subprocess
import tempfile

from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.expr import make_expr_group
from taichi.lang.util import get_clangpp


class SourceBuilder:
    def __init__(self):
        self.bc = None
        self.so = None
        self.mode = None
        self.td = None

        def cleanup():
            if self.td is not None:
                shutil.rmtree(self.td)

        atexit.register(cleanup)

    @classmethod
    def from_file(cls, filename, compile_fn=None, _temp_dir=None):
        self = cls()
        self.td = _temp_dir
        if self.td is None:
            self.td = tempfile.mkdtemp()

        if filename.endswith((".cpp", ".c", ".cc")):
            if impl.current_cfg().arch not in [_ti_core.Arch.x64, _ti_core.Arch.cuda]:
                raise TaichiSyntaxError("Unsupported arch for external function call")
            if compile_fn is None:

                def compile_fn_impl(filename):
                    if impl.current_cfg().arch == _ti_core.Arch.x64:
                        subprocess.call(
                            get_clangpp() + " -flto -c " + filename + " -o " + os.path.join(self.td, "source.bc"),
                            shell=True,
                        )
                    else:
                        subprocess.call(
                            get_clangpp()
                            + " -flto -c "
                            + filename
                            + " -o "
                            + os.path.join(self.td, "source.bc")
                            + " -target nvptx64-nvidia-cuda",
                            shell=True,
                        )
                    return os.path.join(self.td, "source.bc")

                compile_fn = compile_fn_impl
            self.bc = compile_fn(filename)
            self.mode = "bc"
        elif filename.endswith(".cu"):
            if impl.current_cfg().arch not in [_ti_core.Arch.cuda]:
                raise TaichiSyntaxError("Unsupported arch for external function call")
            if compile_fn is None:
                shutil.copy(filename, os.path.join(self.td, "source.cu"))

                def compile_fn_impl(filename):
                    # Cannot use -o to specify multiple output files
                    subprocess.call(
                        get_clangpp()
                        + " "
                        + os.path.join(self.td, "source.cu")
                        + " -c -emit-llvm -std=c++17 --cuda-gpu-arch=sm_50 -nocudalib",
                        cwd=self.td,
                        shell=True,
                    )
                    return os.path.join(self.td, "source-cuda-nvptx64-nvidia-cuda-sm_50.bc")

                compile_fn = compile_fn_impl
            self.bc = compile_fn(filename)
            self.mode = "bc"
        elif filename.endswith((".so", ".dylib", ".dll")):
            if impl.current_cfg().arch not in [_ti_core.Arch.x64]:
                raise TaichiSyntaxError("Unsupported arch for external function call")
            self.so = ctypes.CDLL(filename)
            self.mode = "so"
        elif filename.endswith(".ll"):
            if impl.current_cfg().arch not in [_ti_core.Arch.x64, _ti_core.Arch.cuda]:
                raise TaichiSyntaxError("Unsupported arch for external function call")
            subprocess.call(
                "llvm-as " + filename + " -o " + os.path.join(self.td, "source.bc"),
                shell=True,
            )
            self.bc = os.path.join(self.td, "source.bc")
            self.mode = "bc"
        elif filename.endswith(".bc"):
            if impl.current_cfg().arch not in [_ti_core.Arch.x64, _ti_core.Arch.cuda]:
                raise TaichiSyntaxError("Unsupported arch for external function call")
            self.bc = filename
            self.mode = "bc"
        else:
            raise TaichiSyntaxError("Unsupported file type for external function call.")
        return self

    @classmethod
    def from_source(cls, source_code, compile_fn=None):
        if impl.current_cfg().arch not in [_ti_core.Arch.x64, _ti_core.Arch.cuda]:
            raise TaichiSyntaxError("Unsupported arch for external function call")
        _temp_dir = tempfile.mkdtemp()
        _temp_source = os.path.join(_temp_dir, "_temp_source.cpp")
        with open(_temp_source, "w") as f:
            f.write(source_code)
        return SourceBuilder.from_file(_temp_source, compile_fn, _temp_dir)

    def __getattr__(self, item):
        def bitcode_func_call_wrapper(*args):
            impl.get_runtime().compiling_callable.ast_builder().insert_external_func_call(
                0,
                "",
                self.bc,
                item,
                make_expr_group(args),
                make_expr_group([]),
                _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )

        if self.mode == "bc":
            return bitcode_func_call_wrapper

        def external_func_call_wrapper(args=[], outputs=[]):
            func_addr = ctypes.cast(self.so.__getattr__(item), ctypes.c_void_p).value
            impl.get_runtime().compiling_callable.ast_builder().insert_external_func_call(
                func_addr,
                "",
                "",
                "",
                make_expr_group(args),
                make_expr_group(outputs),
                _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )

        if self.mode == "so":
            return external_func_call_wrapper

        raise TaichiSyntaxError("Error occurs when calling external function.")


__all__ = []
