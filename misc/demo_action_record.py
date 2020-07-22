import taichi as ti
ti.core.start_recording('record.yml')
ti.init(ti.cc, log_level=ti.DEBUG)

@ti.kernel
def func():
    print(233)


func()
