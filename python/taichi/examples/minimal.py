import taichi as ti

ti.init()

pack_type = ti.types.argpack(a=ti.i32)
pack_type2 = ti.types.argpack(a=ti.i32, b=ti.types.ndarray(dtype=ti.math.vec3, ndim=2))
print(pack_type2.dtype)

@ti.kernel
def p() -> ti.f32:
    print(42)
    return 40 + 2


print(p())
