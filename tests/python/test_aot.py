import json
import os
import tempfile

import taichi as ti


@ti.test(arch=ti.cc)
def test_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorded_file = os.path.join(tmpdir, 'record.yml')
        ti.aot.start_recording(recorded_file)

        loss = ti.field(float, (), needs_grad=True)
        x = ti.field(float, 233, needs_grad=True)

        @ti.kernel
        def compute_loss():
            for i in x:
                loss[None] += x[i]**2

        compute_loss()
        ti.aot.stop_recording()

        assert os.path.exists(recorded_file)

        # Make sure kernel info is in the file
        with open(recorded_file, 'r') as f:
            assert 'compute_loss' in ''.join(f.readlines())


@ti.test(arch=ti.opengl)
def test_save():
    density = ti.field(float, shape=(4, 4))

    @ti.kernel
    def init():
        for i, j in density:
            density[i, j] = 1

    @ti.kernel
    def foo(n: ti.template()):
        for i in range(n):
            density[0, 0] += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module(ti.opengl)
        m.add_field('density', density)
        m.add_kernel(init)
        with m.add_kernel_template(foo) as kt:
            kt.instantiate(n=6)
            kt.instantiate(n=8)
        filename = 'taichi_aot_example'
        m.save(tmpdir, filename)
        with open(os.path.join(tmpdir,
                               f'{filename}_metadata.json')) as json_file:
            json.load(json_file)
