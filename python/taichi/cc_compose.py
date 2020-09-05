import yaml
import warnings
import sys


class Composer:
    def __init__(self, entries, fout, hdrout, emscripten=False):
        self.fout = fout
        self.hdrout = hdrout
        self.entries = entries
        self.emscripten = emscripten
        self.launches = []

    def emit(self, line):
        print(line, file=self.fout)

    def emit_header(self, line):
        print(line, file=self.hdrout)

    def run(self):
        if self.emscripten:
            self.emit('#include <emscripten.h>')

        for e in self.entries:
            action = e['action']
            func = getattr(self, 'do_' + action, self.do_unknown)
            func(e)

    def do_unknown(self, e):
        pass

    def do_compile_runtime(self, e):
        header = e['runtime_header']
        source = e['runtime_source']

        self.emit(header)
        self.emit_header(header)
        self.emit(source)
        self.emit('')

    def do_compile_layout(self, e):
        source = e['layout_source']

        self.emit(source)
        self.emit_header(source)
        if self.emscripten:
            self.emit('EMSCRIPTEN_KEEPALIVE')
        self.emit('struct Ti_S0root Ti_root;')
        self.emit('')

    def do_allocate_buffer(self, e):
        root_size = e['root_size']
        gtmp_size = e['gtmp_size']
        extr_size = 4 * 1024 * 1024  # pinpoint: 4 MB

        if self.emscripten:
            self.emit('EMSCRIPTEN_KEEPALIVE')
            self.emit(f'Ti_i8 Ti_extr[{extr_size}];')

        self.emit(f'Ti_i8 Ti_gtmp[{gtmp_size}];')

        if self.emscripten:
            self.emit('EMSCRIPTEN_KEEPALIVE')

        self.emit('union Ti_BitCast Ti_args[8];')
        if self.emscripten:
            self.emit('EMSCRIPTEN_KEEPALIVE')
        self.emit('Ti_i32 Ti_earg[8 * 8];')
        self.emit('')

        if self.emscripten:
            self.emit('EMSCRIPTEN_KEEPALIVE')
        self.emit_header('extern struct Ti_Context Ti_ctx;')
        self.emit_header('')
        self.emit('struct Ti_Context Ti_ctx = {')
        self.emit('  &Ti_root, Ti_gtmp, Ti_args, Ti_earg,')
        self.emit('};')
        self.emit('')

    def do_compile_kernel(self, e):
        name = e['kernel_name']
        source = e['kernel_source']

        if self.emscripten:
            self.emit('EMSCRIPTEN_KEEPALIVE')
        self.emit(source)
        self.emit('')
        declaration = source.split('{', 1)[0].strip()
        self.emit_header(f'extern {declaration};')

    def do_launch_kernel(self, e):
        self.launches.append(e)


def main(fin_name, fout_name, hdrout_name, emscripten=False):
    with open(fin_name, 'r') as fin:
        warnings.filterwarnings('ignore')
        obj = yaml.load(fin)

    with open(hdrout_name, 'w') as hdrout:
        with open(fout_name, 'w') as fout:
            comp = Composer(obj, fout, hdrout, emscripten)
            comp.run()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], len(sys.argv) > 4)
