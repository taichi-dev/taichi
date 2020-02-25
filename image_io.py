import taichi as ti

pixel = ti.var(ti.u8, shape=(128, 128, 3))

@ti.kernel
def paint():
  for I in ti.grouped(pixel):
    pixel[I] = ti.random() * 255

paint()
ti.imshow(pixel, 'Random Generated')
