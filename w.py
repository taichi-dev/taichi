import taichi as ti

ti.init(print_ir=True, advanced_optimization=False, print_preprocessed=True)

@ti.kernel
def func():
    sum = 0
    for i in ti.static(range(5)):
        sum += i
    print(sum)

func()
