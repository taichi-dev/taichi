import taichi as ti
import taichi_glsl as tl


m = ti.var(ti.f32, ())

N = 32

loss = ti.var(ti.f32, 
