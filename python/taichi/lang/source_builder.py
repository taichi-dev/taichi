import atexit
import ctypes
import os
import tempfile

from taichi.lang.exception import TaichiSyntaxError
from taichi.lang.ops import asm, bitcode_func_call, external_func_call
from taichi.lang.util import get_clangpp, has_clangpp


class SourceBuilder():
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
            if self.mode == 'bc':
                os.remove(self.source_file)
                os.remove(self.compiled_file)
                os.removedirs(self.td)
            if self.mode == 'so':
                os.remove(self.source_file)
                os.remove(self.compiled_file)
                os.removedirs(self.td)

        atexit.register(cleanup)

    def __call__(self, inputs=[], outputs=[]):
        if self.mode == 'asm':
            asm(self.asm, inputs, outputs)
            return
        raise TaichiSyntaxError('Error occurs when calling external function.')

    def __getattr__(self, item):
        def bitcode_func_call_wrapper(*args):
            bitcode_func_call(self.bc, item, *args)

        if self.mode == 'bc':
            return bitcode_func_call_wrapper

        def external_func_call_wrapper(args=[], outputs=[]):
            external_func_call(self.so.__getattr__(item), args, outputs)

        if self.mode == 'so':
            return external_func_call_wrapper

        raise TaichiSyntaxError('Error occurs when calling external function.')
