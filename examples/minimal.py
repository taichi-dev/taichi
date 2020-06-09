import taichi as ti

ti.init()

@ti.kernel
def p():
    pass

print(1)
p()
print(2)