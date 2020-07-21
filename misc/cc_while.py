import taichi as ti

ti.init(arch=ti.cc, log_level=ti.DEBUG, debug=True)


@ti.kernel
def func():
    t = 0
    while t < 5:
        t += 1
        break
    print(t)


func()
