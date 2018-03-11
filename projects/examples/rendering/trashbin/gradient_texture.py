import taichi as tc

if __name__ == '__main__':
  while True:
    for i in range(100):
      envmap_texture = tc.Texture(
          'spherical_gradient',
          inside_val=(1, 1, 10, 1),
          outside_val=(0.5, 0.5, 0.5, 1),
          angle=3,
          sharpness=5)
      envmap_texture.show(res=(512, 512))
