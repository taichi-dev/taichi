from math import pi

from taichi._lib import core as _ti_core
from taichi.lang.matrix import Vector

from .utils import check_ggui_availability, euler_to_vec, vec_to_euler


class Camera:
    """The Camera class.

    You should also manually set the camera parameters like `camera.position`,
    `camera.lookat`, `camera.up`, etc. The default settings may not work for
    your scene.

    Example::

        >>> scene = ti.ui.Scene()  # assume you have a scene
        >>>
        >>> camera = ti.ui.Camera()
        >>> camera.position(1, 1, 1)  # set camera position
        >>> camera.lookat(0, 0, 0)  # set camera lookat
        >>> camera.up(0, 1, 0)  # set camera up vector
        >>> scene.set_camera(camera)
        >>>
        >>> # you can also control camera movement in a window
        >>> window = ti.ui.Window("GGUI Camera", res=(640, 480), vsync=True)
        >>> camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    """
    def __init__(self):
        check_ggui_availability()
        self.ptr = _ti_core.PyCamera()

        self.position(0.0, 0.0, 0.0)
        self.lookat(0.0, 0.0, 1.0)
        self.up(0.0, 1.0, 0.0)

        # used for tracking user inputs
        self.last_mouse_x = None
        self.last_mouse_y = None

    def position(self, x, y, z):
        """Set the camera position.

        Args:
            args (:mod:`taichi.types.primitive_types`): 3D coordinates.

        Example::

            >>> camera.position(1, 1, 1)
        """
        self.curr_position = Vector([x, y, z])
        self.ptr.position(x, y, z)

    def lookat(self, x, y, z):
        """Set the camera lookat.

        Args:
            args (:mod:`taichi.types.primitive_types`): 3D coordinates.

        Example::

            >>> camera.lookat(0, 0, 0)
        """
        self.curr_lookat = Vector([x, y, z])
        self.ptr.lookat(x, y, z)

    def up(self, x, y, z):
        """Set the camera up vector.

        Args:
            args (:mod:`taichi.types.primitive_types`): 3D coordinates.

        Example::

            >>> camera.up(0, 1, 0)
        """
        self.curr_up = Vector([x, y, z])
        self.ptr.up(x, y, z)

    def projection_mode(self, mode):
        """Camera projection mode, 0 for perspective and 1 for orthogonal.
        """
        self.ptr.projection_mode(mode)

    def fov(self, fov):
        """Set the camera fov angle (field of view) in degrees.

        Args:
            fov (:mod:`taichi.types.primitive_types`): Angle in range (0, 180).

        Example::

            >>> camera.fov(45)
        """
        self.ptr.fov(fov)

    def left(self, left):
        """Set the offset of the left clipping plane in camera frustum.

        Args:
            left (:mod:`taichi.types.primitive_types`): \
                offset of the left clipping plane.

        Example::

            >>> camera.left(-1.0)
        """
        self.ptr.left(left)

    def right(self, right):
        """Set the offset of the right clipping plane in camera frustum.

        Args:
            right (:mod:`taichi.types.primitive_types`): \
                offset of the right clipping plane.

        Example::

            >>> camera.right(1.0)
        """
        self.ptr.right(right)

    def top(self, top):
        """Set the offset of the top clipping plane in camera frustum.

        Args:
            top (:mod:`taichi.types.primitive_types`): \
                offset of the top clipping plane.

        Example::

            >>> camera.top(-1.0)
        """
        self.ptr.top(top)

    def bottom(self, bottom):
        """Set the offset of the bottom clipping plane in camera frustum.

        Args:
            bottom (:mod:`taichi.types.primitive_types`): \
                offset of the bottom clipping plane.

        Example::

            >>> camera.bottom(1.0)
        """
        self.ptr.bottom(bottom)

    def z_near(self, z_near):
        """Set the offset of the near clipping plane in camera frustum.

        Args:
            near (:mod:`taichi.types.primitive_types`): \
                offset of the near clipping plane.

        Example::

            >>> camera.near(0.1)
        """
        self.ptr.z_near(z_near)

    def z_far(self, z_far):
        """Set the offset of the far clipping plane in camera frustum.

        Args:
            far (:mod:`taichi.types.primitive_types`): \
                offset of the far clipping plane.

        Example::

            >>> camera.left(1000.0)
        """
        self.ptr.z_far(z_far)

    def get_view_matrix(self):
        """Get the view matrix(in row major) of the camera.

        Example::

            >>> camera.get_view_matrix()
        """
        return self.ptr.get_view_matrix()

    def get_projection_matrix(self, aspect):
        """Get the projection matrix(in row major) of the camera.

        Args:
            aspect (:mod:`taichi.types.primitive_types`): \
                aspect ratio of the camera

        Example::

            >>> camera.get_projection_matrix(1080/720)
        """
        return self.ptr.get_projection_matrix(aspect)

    def track_user_inputs(self,
                          window,
                          movement_speed=1.0,
                          yaw_speed=2,
                          pitch_speed=2,
                          hold_key=None):
        """Move the camera according to user inputs.
        Press `w`, `s`, `a`, `d`, `e`, `q` to move the camera
        `formard`, `back`, `left`, `right`, `head up`, `head down`, accordingly.

        Args:

            window (:class:`~taichi.ui.Window`): a windown instance.
            movement_speed (:mod:`~taichi.types.primitive_types`): camera movement speed.
            yaw_speed (:mod:`~taichi.types.primitive_types`): speed of changes in yaw angle.
            pitch_speed (:mod:`~taichi.types.primitive_types`): speed of changes in pitch angle.
            hold_key (:mod:`~taichi.ui`): User defined key for holding the camera movement.
        """
        front = (self.curr_lookat - self.curr_position).normalized()
        position_change = Vector([0.0, 0.0, 0.0])
        left = self.curr_up.cross(front)
        up = self.curr_up
        if window.is_pressed('w'):
            position_change = front * movement_speed
        if window.is_pressed('s'):
            position_change = -front * movement_speed
        if window.is_pressed('a'):
            position_change = left * movement_speed
        if window.is_pressed('d'):
            position_change = -left * movement_speed
        if window.is_pressed('e'):
            position_change = up * movement_speed
        if window.is_pressed('q'):
            position_change = -up * movement_speed
        self.position(*(self.curr_position + position_change))
        self.lookat(*(self.curr_lookat + position_change))

        if hold_key is not None:
            if not window.is_pressed(hold_key):
                self.last_mouse_x = None
                self.last_mouse_y = None
                return

        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
        if self.last_mouse_x is None or self.last_mouse_y is None:
            self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
            return

        dx = curr_mouse_x - self.last_mouse_x
        dy = curr_mouse_y - self.last_mouse_y

        yaw, pitch = vec_to_euler(front)
        yaw_speed = 2
        pitch_speed = 2

        yaw -= dx * yaw_speed
        pitch += dy * pitch_speed

        pitch_limit = pi / 2 * 0.99
        if pitch > pitch_limit:
            pitch = pitch_limit
        elif pitch < -pitch_limit:
            pitch = -pitch_limit

        front = euler_to_vec(yaw, pitch)
        self.lookat(*(self.curr_position + front))
        self.up(0, 1, 0)

        self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
