from taichi.dynamics.nbody import NBody

if __name__ == '__main__':
  nbody = NBody(
      num_particles=200000,
      gravitation=-0.0001,
      vel_scale=10,
      delta_t=0.01,
      num_threads=8)
  for i in range(1000):
    nbody.step(0.01)

  nbody.make_video()
