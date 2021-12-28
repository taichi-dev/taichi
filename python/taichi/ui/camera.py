from math import pi

from taichi.lang.matrix import Vector

from .utils import euler_to_vec, vec_to_euler


class Camera:
    def __init__(self, ptr):
        self.ptr = ptr

        self.position(0.0, 0.0, 0.0)
        self.lookat(0.0, 0.0, 1.0)
        self.up(0.0, 1.0, 0.0)

        # used for tracking user inputs
        self.last_mouse_x = None
        self.last_mouse_y = None

    def position(self, x, y, z):
        self.curr_position = Vector([x, y, z])
        self.ptr.position(x, y, z)

    def lookat(self, x, y, z):
        self.curr_lookat = Vector([x, y, z])
        self.ptr.lookat(x, y, z)

    def up(self, x, y, z):
        self.curr_up = Vector([x, y, z])
        self.ptr.up(x, y, z)

    def projection_mode(self, mode):
        self.ptr.projection_mode(mode)

    def fov(self, fov):
        self.ptr.fov(fov)

    def left(self, left):
        self.ptr.left(left)

    def right(self, right):
        self.ptr.right(right)

    def top(self, top):
        self.ptr.top(top)

    def bottom(self, bottom):
        self.ptr.bottom(bottom)

    def z_near(self, z_near):
        self.ptr.z_near(z_near)

    def z_near(self, z_far):
        self.ptr.z_far(z_far)

    # move the camera according to user inputs, FPS game style.
    def track_user_inputs(self,
                          window,
                          movement_speed=1.0,
                          yaw_speed=2,
                          pitch_speed=2,
                          hold_key=None):
        front = (self.curr_lookat - self.curr_position).normalized()
        position_change = Vector([0.0, 0.0, 0.0])
        left = self.curr_up.cross(front)
        if window.is_pressed('w'):
            position_change = front * movement_speed
        if window.is_pressed('s'):
            position_change = -front * movement_speed
        if window.is_pressed('a'):
            position_change = left * movement_speed
        if window.is_pressed('d'):
            position_change = -left * movement_speed
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
