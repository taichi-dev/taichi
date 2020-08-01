import yaml
import warnings


class Composer:
    def __init__(self, entries):
        self.entries = entries
        self.launches = []

    def run(self):
        for e in self.entries:
            action = e['action']
            func = getattr(self, 'do_' + action, self.do_unknown)
            func(e)

    def do_unknown(self, e):
        print(f"Unknown action type: {e['action']}")

    def do_compile_runtime(self, e):
        header = e['runtime_header']
        source = e['runtime_source']

        print(header)
        print(source)

    def do_compile_layout(self, e):
        source = e['layout_source']

        print(source)
        print('struct Ti_S0root Ti_root;')
        print('')

    def do_allocate_buffer(self, e):
        root_size = e['root_size']
        gtmp_size = e['gtmp_size']

        print(f'Ti_i8 Ti_gtmp[{gtmp_size}];')
        print('union Ti_BitCast Ti_args[8];')
        print('Ti_i32 Ti_earg[8 * 8];')
        print('')
        print('struct Ti_Context Ti_ctx = {')
        print('  &Ti_root, Ti_gtmp, Ti_args, Ti_earg,')
        print('};')
        print('')

    def do_compile_kernel(self, e):
        name = e['kernel_name']
        source = e['kernel_source']

        print(source)
        print('')

    def do_launch_kernel(self, e):
        self.launches.append(e)



if __name__ == '__main__':
    with open('record.yml') as fin:
        warnings.filterwarnings('ignore')
        obj = yaml.load(fin)

    comp = Composer(obj)
    comp.run()
