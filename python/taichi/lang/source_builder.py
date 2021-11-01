import atexit
import ctypes
import os
import tempfile
import shutil
import subprocess

from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.util import get_clangpp, has_clangpp
from taichi.lang.expr import make_expr_group
from taichi.lang import impl
from taichi.core.util import ti_core as _ti_core

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
        if filename.endswith((".cpp", ".c", ".cc")):
            assert impl.current_cfg().arch in [_ti_core.Arch.x64, _ti_core.Arch.cuda]
            if compile_fn is None:
                if self.td is None:
                    self.td = tempfile.mkdtemp()
                def compile_fn_impl(filename):
                    if impl.current_cfg().arch == _ti_core.Arch.x64:
                        subprocess.call(get_clangpp() + ' -flto -c ' + filename + ' -o ' + os.path.join(self.td, 'source.bc'), shell=True)
                    else:
                        subprocess.call(get_clangpp() + ' -flto -c ' + filename + ' -o ' + os.path.join(self.td, 'source.bc') + ' -target nvptx64-nvidia-cuda', shell=True)
                    return os.path.join(self.td, 'source.bc')
                compile_fn = compile_fn_impl
            self.bc = compile_fn(filename)
            self.mode = 'bc'
        elif filename.endswith(".cu"):
            assert impl.current_cfg().arch in [_ti_core.Arch.cuda]
            if compile_fn is None:
                if self.td is None:
                    self.td = tempfile.mkdtemp()
                shutil.copy(filename, os.path.join(self.td, 'source.cu'))
                def compile_fn_impl(filename):
                    # Cannot use -o to specify multiple output files
                    subprocess.call(get_clangpp() + ' ' + os.path.join(self.td, 'source.cu') + ' -S -emit-llvm -std=c++17 --cuda-gpu-arch=sm_50 -nocudalib', cwd=self.td, shell=True)
                    subprocess.call('llvm-as ' + os.path.join(self.td, 'source-cuda-nvptx64-nvidia-cuda-sm_50.ll'), cwd=self.td, shell=True)
                    return os.path.join(self.td, 'source-cuda-nvptx64-nvidia-cuda-sm_50.bc')
                compile_fn = compile_fn_impl
            self.bc = compile_fn(filename)
            self.mode = 'bc'
        elif filename.endswith((".so", ".dylib", ".dll")):
            assert impl.current_cfg().arch in [_ti_core.Arch.x64]
            self.so = ctypes.CDLL(filename)
            self.mode = 'so'
        elif filename.endswith(".ll"):
            assert impl.current_cfg().arch in [_ti_core.Arch.x64, _ti_core.Arch.cuda]
            if self.td is None:
                self.td = tempfile.mkdtemp()
            subprocess.call('llvm-as ' + filename + ' -o ' + os.path.join(self.td, 'source.bc'), shell=True)
            self.bc = os.path.join(self.td, 'source.bc')
            self.mode = 'bc'
        elif filename.endswith(".bc"):
            assert impl.current_cfg().arch in [_ti_core.Arch.x64, _ti_core.Arch.cuda]
            self.bc = filename
            self.mode = 'bc'
        else:
            raise TaichiSyntaxError('Unsupported file type for external function call.')
        return self

    @classmethod
    def from_source(cls, source_code, compile_fn=None):
        assert impl.current_cfg().arch in [_ti_core.Arch.x64, _ti_core.Arch.cuda]
        _temp_dir = tempfile.mkdtemp()
        _temp_source = os.path.join(_temp_dir, '_temp_source.cpp')
        with open(_temp_source, 'w') as f:
            f.write(source_code)
        return SourceBuilder.from_file(_temp_source, compile_fn, _temp_dir)

    def __getattr__(self, item):
        def bitcode_func_call_wrapper(*args):
            _ti_core.insert_external_func_call(0, '', self.bc, item,
                                            make_expr_group(args),
                                            make_expr_group([]))


        if self.mode == 'bc':
            return bitcode_func_call_wrapper

        def external_func_call_wrapper(args=[], outputs=[]):
            func_addr = ctypes.cast(self.so.__getattr__(item), ctypes.c_void_p).value
            _ti_core.insert_external_func_call(func_addr, '', '', '',
                                            make_expr_group(args),
                                            make_expr_group(outputs))


        if self.mode == 'so':
            return external_func_call_wrapper

        raise TaichiSyntaxError('Error occurs when calling external function.')
