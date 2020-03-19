import numbers
import numpy as np


class GUI:
    class Event:
        pass

    SHIFT = 'Shift'
    ALT = 'Alt'
    CTRL = 'Control'
    ESCAPE = 'Escape'
    RETURN = 'Return'
    TAB = 'Tab'
    BACKSPACE = 'BackSpace'
    SPACE = ' '
    UP = 'Up'
    DOWN = 'Down'
    LEFT = 'Left'
    RIGHT = 'Right'
    CAPSLOCK = 'Caps_Lock'
    MOTION = 'Motion'
    LMB = 'LMB'
    MMB = 'MMB'
    RMB = 'RMB'
    RELEASE = False
    PRESS = True

    def __init__(self, name, res=512, background_color=0x0):
        import taichi as ti
        self.name = name
        if isinstance(res, numbers.Number):
            res = (res, res)
        self.res = res
        self.core = ti.core.GUI(name, ti.veci(*res))
        self.canvas = self.core.get_canvas()
        self.background_color = background_color
        self.key_pressed = set()
        self.clear()
        if ti.core.get_current_program():
            self.core.set_profiler(
                ti.core.get_current_program().get_profiler())

    def clear(self, color=None):
        if color is None:
            color = self.background_color
        self.canvas.clear(color)

    def set_image(self, img):
        import numpy as np
        from .image import cook_image
        img = cook_image(img)
        if img.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            img = img.astype(np.float32) * (1 / np.iinfo(img.dtype).max)
        elif img.dtype in [np.float32, np.float64]:
            img = img.astype(np.float32)
        else:
            raise ValueError(
                f'Data type {img.dtype} not supported in GUI.set_image')
        if len(img.shape) == 2:
            img = img[..., None]
        if img.shape[2] == 1:
            img = img + np.zeros(shape=(1, 1, 4))
        if img.shape[2] == 3:
            img = np.concatenate([
                img,
                np.zeros(shape=(img.shape[0], img.shape[1], 1),
                         dtype=np.float32)
            ],
                                 axis=2)
        img = img.astype(np.float32)
        assert img.shape[:
                         2] == self.res, "Image resolution does not match GUI resolution"
        self.core.set_img(np.ascontiguousarray(img).ctypes.data)

    def circle(self, pos, color, radius=1):
        self.canvas.circle_single(pos[0], pos[1], color, radius)

    def circles(self, pos, color=0xFFFFFF, radius=1):
        n = pos.shape[0]
        if len(pos.shape) == 3:
            assert pos.shape[2] == 1
            pos = pos[:, :, 0]

        assert pos.shape == (n, 2)
        pos = np.ascontiguousarray(pos.astype(np.float32))
        pos = int(pos.ctypes.data)

        if isinstance(color, np.ndarray):
            assert color.shape == (n, )
            color = np.ascontiguousarray(color.astype(np.uint32))
            color_array = int(color.ctypes.data)
            color_single = 0
        elif isinstance(color, int):
            color_array = 0
            color_single = color
        else:
            raise ValueError(
                'Color must be an ndarray or int (e.g., 0x956333)')

        if isinstance(radius, np.ndarray):
            assert radius.shape == (n, )
            radius = np.ascontiguousarray(radius.astype(np.float32))
            radius_array = int(radius.ctypes.data)
            radius_single = 0
        elif isinstance(radius, numbers.Number):
            radius_array = 0
            radius_single = radius
        else:
            raise ValueError('Radius must be an ndarray or float (e.g., 0.4)')

        self.canvas.circles_batched(n, pos, color_single, color_array,
                                    radius_single, radius_array)

    def triangle(self, a, b, c, color=0xFFFFFF):
        self.canvas.triangle_single(a[0], a[1], b[0], b[1], c[0], c[1], color)

    def line(self, begin, end, radius, color):
        self.canvas.path_single(begin[0], begin[1], end[0], end[1], color,
                                radius)

    def show(self, file=None):
        self.core.update()
        if file:
            self.core.screenshot(file)
        self.clear(self.background_color)

    def has_key_event(self):
        return self.core.has_key_event()

    def get_key_event(self):
        self.core.wait_key_event()
        e = GUI.Event()
        e.key = self.core.get_key_event_head_key()
        e.type = self.core.get_key_event_head_type()
        e.pos = self.core.get_key_event_head_pos()
        e.modifier = []
        for mod in ['Shift', 'Alt', 'Control']:
            if self.is_pressed(mod):
                e.modifier.append(mod)
        if e.type == GUI.PRESS:
            self.key_pressed.add(e.key)
        else:
            self.key_pressed.discard(e.key)
        self.core.pop_key_event_head()
        return e

    def is_pressed(self, *keys):
        for key in keys:
            if key in ['Shift', 'Alt', 'Control']:
                if key + '_L' in self.key_pressed or key + '_R' in self.key_pressed:
                    return True
            elif key in self.key_pressed:
                return True
        else:
            return False

    def get_cursor_pos(self):
        return self.core.get_cursor_pos()

    def wait_key():
        while True:
            key, type = self.get_key_event()
            if type == GUI.PRESS:
                return key

    def has_key_pressed(self):
        if self.has_key_event():
            self.get_key_event()  # pop to update self.key_pressed
        return len(self.key_pressed) != 0


def rgb_to_hex(c):
    to255 = lambda x: min(255, max(0, int(x * 255)))
    return 65536 * to255(c[0]) + 256 * to255(c[1]) + to255(c[2])
