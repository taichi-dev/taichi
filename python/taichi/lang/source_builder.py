import tempfile
import os
import atexit
import ctypes
from taichi.lang.ops import external_func_call, bitcode_func_call, asm
from taichi.lang.exception import TaichiSyntaxError

class SourceBuilder():
    def __init__(self, source, mode='bc'):
        self.mode = mode
        if self.mode == 'bc':
            self.td = tempfile.mkdtemp()
            with open(self.td + '/source.cpp', 'w') as f:
                f.write(source)
            os.system('clang++-11 -flto -c ' + self.td + '/source.cpp' + ' -o ' + self.td + '/source.bc')
            self.bc = self.td + '/source.bc'
        elif self.mode == 'so':
            self.td = tempfile.mkdtemp()
            with open(self.td + '/source.cpp', 'w') as f:
                f.write(source)
            os.system('g++ ' + self.td + '/source.cpp -o ' + self.td + '/source.so' + ' -fPIC -shared')
            self.so = ctypes.CDLL(self.td + '/source.so')
        elif self.mode == 'asm':
            self.asm = source
        else:
            raise TaichiSyntaxError('Only support bc, so or asm modes.')

        def cleanup():
            if self.mode == 'bc':
                os.remove(self.td + '/source.cpp')
                os.remove(self.td + '/source.bc')
                os.removedirs(self.td)
            if self.mode == 'so':
                os.remove(self.td + '/source.cpp')
                os.remove(self.td + '/source.so')
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
