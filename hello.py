import taichi as ti
import traceback
import sys

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

def excepthook(type, value, tb):
    for frame, lineno in traceback.walk_tb(tb):
        if frame.f_locals.get('_taichi_no_traceback'):
            pass
        traceback.print_stack(frame)

sys.excepthook = excepthook

main()
