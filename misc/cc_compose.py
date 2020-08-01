import yaml
import warnings


class Composer:
    def __init__(self, entries, emscripten=False):
        self.entries = entries
        self.emscripten = emscripten
        self.launches = []

    def run(self):
        if self.emscripten:
            print('#include <emscripten.h>')

        for e in self.entries:
            action = e['action']
            func = getattr(self, 'do_' + action, self.do_unknown)
            func(e)

        if self.emscripten:
            print('')
            print('EMSCRIPTEN_KEEPALIVE')
            print('void Ti_set_arg_i32(struct Ti_Context *ti_ctx, ')
            print('                    Ti_i32 index, Ti_i32 value) {')
            print('  ti_ctx->args[index].val_i32 = value;')
            print('}')
            print('')
            print('EMSCRIPTEN_KEEPALIVE')
            print('void Ti_set_arg_i64(struct Ti_Context *ti_ctx, ')
            print('                    Ti_i32 index, Ti_i64 value) {')
            print('  ti_ctx->args[index].val_i64 = value;')
            print('}')
            print('')
            print('EMSCRIPTEN_KEEPALIVE')
            print('void Ti_set_arg_f32(struct Ti_Context *ti_ctx, ')
            print('                    Ti_i32 index, Ti_f32 value) {')
            print('  ti_ctx->args[index].val_f32 = value;')
            print('}')
            print('')
            print('EMSCRIPTEN_KEEPALIVE')
            print('void Ti_set_arg_f64(struct Ti_Context *ti_ctx, ')
            print('                    Ti_i32 index, Ti_f64 value) {')
            print('  ti_ctx->args[index].val_f64 = value;')
            print('}')
            print('')
            print('EMSCRIPTEN_KEEPALIVE')
            print('void Ti_set_arg_ptr(struct Ti_Context *ti_ctx, ')
            print('                    Ti_i32 index, void *ptr) {')
            print('  ti_ctx->args[index].ptr_void = ptr;')
            print('}')
            print('')
            print('')

    def do_unknown(self, e):
        print(f"Unknown action type: {e['action']}")

    def do_compile_runtime(self, e):
        header = e['runtime_header']
        source = e['runtime_source']

        print(header)
        print(source)
        print('')

    def do_compile_layout(self, e):
        source = e['layout_source']

        print(source)
        if self.emscripten:
            print('EMSCRIPTEN_KEEPALIVE')
        print('struct Ti_S0root Ti_root;')
        print('')

    def do_allocate_buffer(self, e):
        root_size = e['root_size']
        gtmp_size = e['gtmp_size']
        extr_size = 4 * 1024 * 1024  # pinpoint: 4 MB

        if self.emscripten:
            print('EMSCRIPTEN_KEEPALIVE')
            print(f'Ti_i8 Ti_extr[{extr_size}];')

        print(f'Ti_i8 Ti_gtmp[{gtmp_size}];')

        if self.emscripten:
            print('EMSCRIPTEN_KEEPALIVE')

        print('union Ti_BitCast Ti_args[8];')
        if self.emscripten:
            print('EMSCRIPTEN_KEEPALIVE')
        print('Ti_i32 Ti_earg[8 * 8];')
        print('')

        if self.emscripten:
            print('EMSCRIPTEN_KEEPALIVE')
        print('struct Ti_Context Ti_ctx = {')
        print('  &Ti_root, Ti_gtmp, Ti_args, Ti_earg,')
        print('};')
        print('')

    def do_compile_kernel(self, e):
        name = e['kernel_name']
        source = e['kernel_source']

        if self.emscripten:
            print('EMSCRIPTEN_KEEPALIVE')
        print(source)
        print('')

    def do_launch_kernel(self, e):
        self.launches.append(e)



if __name__ == '__main__':
    with open('record.yml') as fin:
        warnings.filterwarnings('ignore')
        obj = yaml.load(fin)

    comp = Composer(obj, emscripten=True)
    comp.run()
