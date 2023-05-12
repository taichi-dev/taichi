import random

import matplotlib.pyplot as plt
import numpy as np

import taichi as ti

ti.init(arch=ti.cpu)

number_coeffs = 4
learning_rate = 1e-4

N = 32
x, y = ti.field(ti.f32, shape=N, needs_grad=True), ti.field(ti.f32, shape=N, needs_grad=True)
coeffs = ti.field(ti.f32, shape=number_coeffs, needs_grad=True)
loss = ti.field(ti.f32, shape=(), needs_grad=True)


@ti.kernel
def regress():
    for i in x:
        v = x[i]
        est = 0.0
        for j in ti.static(range(number_coeffs)):
            est += coeffs[j] * (v**j)
        loss[None] += 0.5 * (y[i] - est) ** 2


@ti.kernel
def update():
    for i in ti.static(range(number_coeffs)):
        coeffs[i] -= learning_rate * coeffs.grad[i]


xs = []
ys = []


def initialize():
    for i in range(N):
        v = random.random() * 5 - 2.5
        xs.append(v)
        x[i] = v
        y[i] = (v - 1) * (v - 2) * (v + 2) + random.random() - 0.5

    regress()

    print("y")
    for i in range(N):
        y.grad[i] = 1
        ys.append(y[i])
    print()


def regress_raw():
    use_tape = True

    for i in range(1000):
        if use_tape:
            with ti.ad.Tape(loss=loss):
                regress()
        else:
            ti.ad.clear_all_gradients()
            loss[None] = 0
            loss.grad[None] = 1
            regress()
            regress.grad()
        print("Loss =", loss[None])
        update()
        for j in range(number_coeffs):
            print(coeffs[j], end=", ")
        print()


def draw():
    curve_xs = np.arange(-2.5, 2.5, 0.01)
    curve_ys = curve_xs * 0
    for i in range(number_coeffs):
        curve_ys += coeffs[i] * np.power(curve_xs, i)

    plt.title("Nonlinear Regression with Gradient Descent (3rd order polynomial)")
    ax = plt.gca()
    ax.scatter(xs, ys, label="data", color="r")
    ax.plot(curve_xs, curve_ys, label="fitted")
    ax.legend()
    ax.grid(True)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")
    plt.show()


def main():
    initialize()
    regress_raw()
    draw()


if __name__ == "__main__":
    main()
