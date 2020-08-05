import yaml
import warnings
import sys


class Composer:
    def __init__(self, fout, entries, emscripten=False):
        self.fout = fout
        self.entries = entries
        self.emscripten = emscripten
        self.launches = []

    def emit(self, line):
        print(line, file=self.fout)

    def run(self):
        if self.emscripten:
            self.emit('#include <emscripten.h>')

        for e in self.entries:
            action = e['action']
            func = getattr(self, 'do_' + action, self.do_unknown)
            func(e)

    def do_unknown(self, e):
        self.emit(f"Unknown action type: {e['action']}")

    def do_compile_runtime(self, e):
        header = e['runtime_header']
        source = e['runtime_source']

        self.emit(header)
        self.emit(source)
        self.emit('')

    def do_compile_layout(self, e):
        source = e['layout_source']

        self.emit(source)
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

    def do_launch_kernel(self, e):
        self.launches.append(e)


def main(fin_name, fout_name, emscripten=False):
    with open(fin_name, 'r') as fin:
        warnings.filterwarnings('ignore')
        obj = yaml.load(fin)

    with open(fout_name, 'w') as fout:
        comp = Composer(fout, obj, emscripten)
        comp.run()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3)
