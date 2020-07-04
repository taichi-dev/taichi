import taichi as ti

@ti.func
def func3():
    ti.static_assert(1 + 1 == 3)

@ti.func
def func2():
    func3()

@ti.func
def func1():
    func2()

@ti.kernel
def func0():
    func1()

ti.enable_excepthook()
func0()


'''Before:
[Taichi] mode=development
[Taichi] preparing sandbox at /tmp/taichi-4pwjc_r7
[Taichi] <dev mode>, llvm 10.0.0, commit c24d6da8, python 3.8.3
Traceback (most recent call last):
  File "misc/w.py", line 20, in <module>
    func0()
  File "/root/taichi/python/taichi/lang/kernel.py", line 553, in wrapped
    return primal(*args, **kwargs)
  File "/root/taichi/python/taichi/lang/kernel.py", line 483, in __call__
    self.materialize(key=key, args=args, arg_features=arg_features)
  File "/root/taichi/python/taichi/lang/kernel.py", line 363, in materialize
    taichi_kernel = taichi_kernel.define(taichi_ast_generator)
  File "/root/taichi/python/taichi/lang/kernel.py", line 360, in taichi_ast_generator
    compiled()
  File "misc/w.py", line 17, in func0
    func1()
  File "/root/taichi/python/taichi/lang/kernel.py", line 38, in decorated
    return fun.__call__(*args)
  File "/root/taichi/python/taichi/lang/kernel.py", line 77, in __call__
    ret = self.compiled(*args)
  File "misc/w.py", line 13, in func1
    func2()
  File "/root/taichi/python/taichi/lang/kernel.py", line 38, in decorated
    return fun.__call__(*args)
  File "/root/taichi/python/taichi/lang/kernel.py", line 77, in __call__
    ret = self.compiled(*args)
  File "misc/w.py", line 9, in func2
    func3()
  File "/root/taichi/python/taichi/lang/kernel.py", line 38, in decorated
    return fun.__call__(*args)
  File "/root/taichi/python/taichi/lang/kernel.py", line 77, in __call__
    ret = self.compiled(*args)
  File "misc/w.py", line 5, in func3
    ti.static_assert(1 + 1 == 3)
  File "/root/taichi/python/taichi/lang/impl.py", line 249, in static_assert
    assert cond
AssertionError
'''


'''After:
[Taichi] mode=development
[Taichi] preparing sandbox at /tmp/taichi-0x77_x75
[Taichi] <dev mode>, llvm 10.0.0, commit c24d6da8, python 3.8.3
========== Taichi Stack Traceback ==========
In <module>() at misc/w.py:20:
--------------------------------------------
def func0():
    func1()

ti.enable_excepthook()
func0()  <--

--------------------------------------------
In func0() at misc/w.py:17:
--------------------------------------------
    func2()

@ti.kernel
def func0():
    func1()  <--

ti.enable_excepthook()
--------------------------------------------
In func1() at misc/w.py:13:
--------------------------------------------
    func3()

@ti.func
def func1():
    func2()  <--

@ti.kernel
--------------------------------------------
In func2() at misc/w.py:9:
--------------------------------------------
    ti.static_assert(1 + 1 == 3)

@ti.func
def func2():
    func3()  <--

@ti.func
--------------------------------------------
In func3() at misc/w.py:5:
--------------------------------------------
import taichi as ti

@ti.func
def func3():
    ti.static_assert(1 + 1 == 3)  <--

@ti.func
--------------------------------------------
AssertionError
''' # and, with colors!!!!!
