import taichi as ti
import math
import matplotlib.pyplot as plt

ti.init()


def main():
    use_cft = True
    use_shared_exp = False
    if use_cft:
        exp = ti.type_factory.custom_int(6, False)
        cit = ti.type_factory.custom_int(13, True)
        cft = ti.type_factory.custom_float(significand_type=cit,
                                           exponent_type=exp,
                                           scale=1)
        x = ti.Vector.field(2, dtype=cft)
        if use_shared_exp:
            ti.root._bit_struct(num_bits=32).place(x, shared_exponent=True)
        else:
            ti.root._bit_struct(num_bits=32).place(x(0))
            ti.root._bit_struct(num_bits=32).place(x(1))
    else:
        x = ti.Vector.field(2, dtype=ti.f32, shape=())

    @ti.func
    def v_at(p):
        return ti.Vector([p[1], -p[0]])

    @ti.kernel
    def advance(dt: ti.f32):
        v_mid = v_at(x[None] + 0.5 * dt * v_at(x[None]))
        x[None] = x[None] + v_mid * dt

    x[None] = [1, 0]
    num_steps = 100
    dt = math.pi * 2 / num_steps
    px = [1]
    py = [0]
    for i in range(num_steps):
        advance(dt)
        px.append(x[None][0])
        py.append(x[None][1])
    plt.plot(px, py)
    plt.show()


main()
