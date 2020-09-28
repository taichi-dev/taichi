import taichi as ti

from util import benchmark_async


# TODO: staggerred grid

@benchmark_async
def advection_2d(scale):
    v = ti.Vector.field(2, dtype=ti.f32)
    
    ti.root.pointer(ti.ij, 128).dense(ti.ij, 8).place(v)
    
    
    
    
    
    
    
    
if __name__ == '__main__':

