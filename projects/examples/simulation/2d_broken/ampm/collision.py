from taichi.misc.util import *
from taichi.two_d import *
import taichi as tc

if __name__ == '__main__':
  scale = 128
  res = (scale, scale)
  frame_dt = 1e-2
  async = False
  gravity = (0, 0)
  if async:
    simulator = MPMSimulator(
        res=res,
        simulation_time=0.4,
        frame_dt=frame_dt,
        base_delta_t=1e-6,
        async=True,
        maximum_delta_t=2e-1,
        debug_input=(128, 7, 0, 0),
        cfl=0.5,
        gravity=gravity,
        show_limits=True)
  else:
    simulator = MPMSimulator(
        res=res,
        simulation_time=0.4,
        frame_dt=frame_dt,
        base_delta_t=1e-3,
        async=False,
        maximum_delta_t=1e-1,
        debug_input=(1024, 4, 0, 0),
        gravity=gravity)

  # simulator.test()

  def event(s):
    w, h = 0.10, 0.05

    def get_polygon(center_x, center_y):
      polygon = [
          Vector(center_x - w, center_y - h),
          Vector(center_x + w, center_y - h),
          Vector(center_x + w, center_y + h),
          Vector(center_x - w, center_y + h),
      ]
      return polygon

    s.add_particles_polygon(
        get_polygon(0.7, 0.53),
        'ep',
        compression=1.0,
        velocity=Vector(-3, 0),
        theta_s=1,
        theta_c=1)
    s.add_particles_polygon(
        get_polygon(0.3, 0.47),
        'ep',
        compression=1.0,
        velocity=Vector(3, 0),
        theta_s=1,
        theta_c=1)

  simulator.add_event(-1, event)

  levelset = simulator.create_levelset()
  bd = -1
  levelset.add_polygon([(bd, bd), (1 - bd, bd), (1 - bd, 1 - bd), (bd, 1 - bd)],
                       True)
  simulator.set_levelset(levelset)
  window = SimulationWindow(
      512,
      simulator,
      color_schemes['snow'],
      levelset_supersampling=2,
      show_images=True,
      substep=False,
      video_output=False,
      need_press=False)
