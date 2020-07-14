from time import sleep
import taichi as ti
ti.init()

def do_something_A():
    sleep(0.01)

def do_something_B():
    sleep(0.1)

ti.profiler.start('A')
do_something_A()
ti.profiler.stop('A')

ti.profiler.start('B')
do_something_B()
ti.profiler.stop('B')

ti.profiler.print()
