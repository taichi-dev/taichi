import taichi as ti

ti.init(arch=ti.cpu, print_ir=True)

n = 4
m = 8

a = ti.field(dtype=ti.i32)
ti.root.dense(ti.ij, (1, 2)).dense(ti.ij, 2).dense(ti.ij, 2).place(a)


@ti.kernel
def fill():
    for i, j in a:
        base = ti.get_addr(a.snode, [0, 0])
        a[i, j] = int(ti.get_addr(a.snode, [i, j]) - base) // 4


def main():
    fill()
    print(a.to_numpy())

    gui = ti.GUI("layout", res=(256, 512), background_color=0xFFFFFF)

    while True:
        for i in range(1, m):
            gui.line(begin=(0, i / m), end=(1, i / m), radius=2, color=0x000000)
        for i in range(1, n):
            gui.line(begin=(i / n, 0), end=(i / n, 1), radius=2, color=0x000000)
        for i in range(n):
            for j in range(m):
                gui.text(
                    f"{a[i, j]}",
                    ((i + 0.3) / n, (j + 0.75) / m),
                    font_size=30,
                    color=0x0,
                )
        gui.show()


if __name__ == "__main__":
    main()
