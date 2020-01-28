import taichi as ti
import time

# TODO: these are not really tests...
@ti.all_archs
def test_basic():
  @ti.kernel
  def test():
    for i in range(10):
      ti.call_internal("do_nothing")
    
  test()


@ti.all_archs
def test_host_polling():
  return
  @ti.kernel
  def test():
    ti.call_internal("refresh_counter")
  
  for i in range(10):
    print('updating tail to', i)
    test()
    time.sleep(0.1)
    
@ti.all_archs
def test_list_manager():
  @ti.kernel
  def test():
    ti.call_internal("test_list_manager")
  
  test()
  test()


@ti.all_archs
def test_node_manager():
  @ti.kernel
  def test():
    ti.call_internal("test_node_allocator")
  
  test()
  test()

