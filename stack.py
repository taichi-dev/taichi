import taichi as ti

@ti.func
def hello1():
    print(a)

@ti.func
def hello2():
    hello1()

@ti.func
def hello3():
    hello2()

@ti.kernel
def main():
    hello3()

main()
