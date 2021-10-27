import atexit
import ctypes
import os
import tempfile

from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.util import get_clangpp, has_clangpp
from taichi.lang.expr import make_expr_group
from taichi.core.util import ti_core as _ti_core

class SourceBuilder:
    def __init__(self, source, mode='bc'):
        self.mode = mode
        if self.mode == 'bc':
            self.td = tempfile.mkdtemp()
            self.source_file = os.path.join(self.td, 'source.cpp')
            self.compiled_file = os.path.join(self.td, 'source.bc')
            with open(self.source_file, 'w') as f:
                f.write(source)
            assert has_clangpp()
            os.system(get_clangpp() + ' -flto -c ' + self.source_file +
                      ' -o ' + self.compiled_file)
            self.bc = self.compiled_file
        elif self.mode == 'so':
            self.td = tempfile.mkdtemp()
            self.source_file = os.path.join(self.td, 'source.cpp')
            self.compiled_file = os.path.join(self.td, 'source.so')
            with open(self.source_file, 'w') as f:
                f.write(source)
            os.system('g++ ' + self.source_file + ' -o ' + self.compiled_file +
                      ' -fPIC -shared')
            self.so = ctypes.CDLL(self.compiled_file)
        elif self.mode == 'asm':
            self.asm = source
        else:
            raise TaichiSyntaxError('Only support bc, so or asm modes.')

        def cleanup():
            if self.mode == 'bc' or self.mode == "so":
                os.remove(self.source_file)
                os.remove(self.compiled_file)
                os.removedirs(self.td)

        atexit.register(cleanup)

    def __call__(self, inputs=[], outputs=[]):
        if self.mode == 'asm':
            _ti_core.insert_external_func_call(0, self.asm, '', '',
                                            make_expr_group(inputs),
                                            make_expr_group(outputs))
            return
        raise TaichiSyntaxError('Error occurs when calling external function.')

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
