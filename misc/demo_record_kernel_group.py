import taichi as ti

ti.aot.start_recording('record.yml')
ti.init(arch=ti.cc)

loss = ti.field(float, (), needs_grad=True)
x = ti.field(float, 233, needs_grad=True)


@ti.kernel
def compute_loss():
    for i in x:
        loss[None] += x[i]**2


@ti.kernel
def do_some_works():
    for i in x:
        x[i] -= x.grad[i]


with ti.aot.RecordKernelGroup('my_substep'):
    x.fill(0)
    with ti.Tape(loss):
        compute_loss()
    do_some_works()
