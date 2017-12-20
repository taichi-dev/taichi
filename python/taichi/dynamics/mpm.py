from taichi.core import tc_core
from .levelset import LevelSet
from taichi.misc.util import *
from taichi.tools.video import VideoManager
from taichi.visual.camera import Camera
from taichi.visual.particle_renderer import ParticleRenderer
from taichi.visual.post_process import LDRDisplay
from taichi.gui.image_viewer import show_image
import taichi as tc
import math
import errno
import sys

class MPM:
  def __init__(self, snapshot_interval=20, **kwargs):
    res = kwargs['res']
    self.frame_dt = kwargs.get('frame_dt', 0.01)
    if 'frame_dt' not in kwargs:
      kwargs['frame_dt'] = self.frame_dt
    self.num_frames = kwargs.get('num_frames', 1000)
    if len(res) == 2:
      self.c = tc_core.create_simulation2('mpm')
      self.Vector = tc_core.Vector2f
      self.Vectori = tc_core.Vector2i
    else:
      self.c = tc.core.create_simulation3('mpm')
      self.Vector = tc_core.Vector3f
      self.Vectori = tc_core.Vector3i
    
    self.snaoshot_interval = snapshot_interval

    if 'task_id' in kwargs:
      self.task_id = kwargs['task_id']
    else:
      self.task_id = sys.argv[0].split('.')[0]
    if 'delta_x' not in kwargs:
      kwargs['delta_x'] = 1.0 / res[0]
    else:
      self.task_id = sys.argv[0].split('.')[0]
      
    print('delta_x = {}'.format(kwargs['delta_x']))
    print('task_id = {}'.format(self.task_id))
    
    self.directory = tc.get_output_path('mpm/' + self.task_id, True)
    self.snapshot_directory = os.path.join(self.directory, 'snapshots')
    self.video_manager = VideoManager(self.directory)
    kwargs['frame_directory'] = self.video_manager.get_frame_directory()
    self.c.initialize(P(**kwargs))
    try:
      os.mkdir(self.directory)
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise
    vis_res = self.c.get_vis_resolution()
    self.video_manager.width = vis_res.x
    self.video_manager.height = vis_res.y
    self.particle_renderer = ParticleRenderer(
        'shadow_map',
        shadow_map_resolution=0.3,
        alpha=0.7,
        shadowing=2,
        ambient_light=0.01,
        light_direction=(1, 1, 0))
    self.res = kwargs['res']
    self.c.frame = 0

    dummy_levelset = self.create_levelset()

    def dummy_levelset_generator(_):
      return dummy_levelset

    self.levelset_generator = dummy_levelset_generator
    self.start_simulation_time = None
    self.simulation_total_time = None
    self.visualize_count = 0
    self.visualize_count_limit = 400000.0

  def add_particles(self, **kwargs):
    return self.c.add_particles(P(**kwargs))

  def update_levelset(self, t0, t1):
    if len(self.res) == 2:
      levelset = tc.core.DynamicLevelSet2D()
    else:
      levelset = tc.core.DynamicLevelSet3D()
    levelset.initialize(t0, t1,
                        self.levelset_generator(t0).levelset,
                        self.levelset_generator(t1).levelset)
    self.c.set_levelset(levelset)

  def set_levelset(self, levelset, is_dynamic_levelset=False):
    if is_dynamic_levelset:
      self.levelset_generator = levelset
    else:

      def levelset_generator(_):
        return levelset

      self.levelset_generator = levelset_generator

  def get_current_time(self):
    return self.c.get_current_time()

  def step(self, step_t, camera=None):
    t = self.c.get_current_time()
    print('* Current t: %.3f' % t)
    self.update_levelset(t, t + step_t)
    T = time.time()
    if not self.start_simulation_time:
      self.start_simulation_time = T
    if not self.simulation_total_time:
      self.simulation_total_time = 0
    self.c.step(step_t)
    self.simulation_total_time += time.time() - T
    print('* Step Time: %.2f [tot: %.2f per frame %.2f]' %
          (time.time() - T, time.time() - self.start_simulation_time,
           self.simulation_total_time / (self.c.frame + 1)))
    image_buffer = tc_core.Array2DVector3(
        Vectori(self.video_manager.width, self.video_manager.height),
        Vector(0, 0, 0.0))
    '''
    particles = self.c.get_render_particles()
    try:
      os.mkdir(self.directory + '/particles')
    except:
      pass
    particles.write(self.directory + '/particles/%05d.bin' % self.c.frame)
    '''
    res = list(map(float, self.res))
    r = res[0]
    if not camera:
      camera = Camera(
          'pinhole',
          origin=(0, r * 0.4, r * 1.4),
          look_at=(0, -r * 0.5, 0),
          up=(0, 1, 0),
          fov=90,
          res=(10, 10))
    if False:
      self.particle_renderer.set_camera(camera)
      self.particle_renderer.render(image_buffer, particles)
      img = image_buffer_to_ndarray(image_buffer)
      img = LDRDisplay(exposure=2.0, adaptive_exposure=False).process(img)
      show_image('Vis', img)
      self.video_manager.write_frame(img)
    self.c.frame += 1

  def get_directory(self):
    return self.directory

  def make_video(self):
    self.video_manager.make_video()

  def create_levelset(self):
    ls = LevelSet(Vectori(self.res), self.Vector(0.0))
    return ls

  def test(self):
    return self.c.test()

  def get_mpi_world_rank(self):
    return self.c.get_mpi_world_rank()

  def visualize(self):
    self.c.visualize()
    self.visualize_count += 1
    if self.visualize_count == int(self.visualize_count_limit):
      self.video_manager.make_video()
      self.visualize_count_limit *= math.sqrt(2)

  def get_debug_information(self):
    return self.c.get_debug_information()

  def clear_output_directory(self):
    frames_dir = os.path.join(self.directory, 'frames')
    taichi.clear_directory_with_suffix(frames_dir, 'json')
    taichi.clear_directory_with_suffix(frames_dir, 'bgeo')
    taichi.clear_directory_with_suffix(frames_dir, 'obj')
    
  def simulate(self, clear_output_directory=False, print_profile_info=False, frame_update=None, update_frequency=1):
    if clear_output_directory:
      self.clear_output_directory()
    for i in range(self.num_frames):
      print('Simulating frame {}'.format(i))
      for k in range(update_frequency):
        if frame_update:
          frame_update(self.get_current_time(), self.frame_dt / update_frequency)
        self.step(self.frame_dt / update_frequency)
      self.visualize()
      if print_profile_info:
        tc.core.print_profile_info()
  
  def add_articulation(self, **kwargs):
    kwargs['action'] = 'add_articulation'
    self.c.general_action(P(**kwargs))
    
  def action(self, **kwargs):
    self.c.general_action(P(**kwargs))
    
  def save(self, fn):
    self.action(action="save", file_name=fn)
    
  def load(self, fn):
    self.action(action="load", file_name=fn)
    
  def get_snapshot_file_name(self, iteration):
    return os.path.join(self.snapshot_directory, '{:04d}.tcb'.format(iteration))
    
