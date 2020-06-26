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
    EXIT = 'WMClose'
    RELEASE = False
    PRESS = True

    def __init__(self, name, res=512, background_color=0x0):
        import taichi as ti
        self.name = name
        if isinstance(res, numbers.Number):
            res = (res, res)
        self.res = res
        # The GUI canvas uses RGBA for storage, therefore we need NxMx4 for an image.
        self.img = np.ascontiguousarray(np.zeros(self.res + (4, ), np.float32))
        self.core = ti.core.GUI(name, ti.veci(*res))
        self.canvas = self.core.get_canvas()
        self.background_color = background_color
        self.key_pressed = set()
        self.event = None
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, type, val, tb):
        self.core = None  # dereference to call GUI::~GUI()

    def clear(self, color=None):
        if color is None:
            color = self.background_color
        self.canvas.clear(color)

    def cook_image(self, img):
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
            img = img + np.zeros((1, 1, 4), np.float32)
        if img.shape[2] == 3:
            zeros = np.zeros((img.shape[0], img.shape[1], 1), np.float32)
            img = np.concatenate([img, zeros], axis=2)

        res = img.shape[:2]
        assert res == self.res, "Image resolution does not match GUI resolution"
        return np.ascontiguousarray(img)

    def set_image(self, img):
        import numpy as np
        import taichi as ti

        if isinstance(img, ti.Expr):
            if ti.core.is_integral(img.data_type()) or len(img.shape) != 2:
                # Images of uint is not optimized by xxx_to_image
                self.img = self.cook_image(img.to_numpy())
            else:
                # Type matched! We can use an optimized copy kernel.
                assert img.shape(
                ) == self.res, "Image resolution does not match GUI resolution"
                from taichi.lang.meta import tensor_to_image
                tensor_to_image(img, self.img)
                ti.sync()

        elif isinstance(img, ti.Matrix):
            if ti.core.is_integral(img.data_type()):
                self.img = self.cook_image(img.to_numpy())
            else:
                # Type matched! We can use an optimized copy kernel.
                assert img.shape(
                ) == self.res, "Image resolution does not match GUI resolution"
                assert img.n in [
                    3, 4
                ], "Only greyscale, RGB or RGBA images are supported in GUI.set_image"
                assert img.m == 1
                from taichi.lang.meta import vector_to_image
                vector_to_image(img, self.img)
                ti.sync()

        elif isinstance(img, np.ndarray):
            self.img = self.cook_image(img)

        else:
            raise ValueError(
                f"GUI.set_image only takes a Taichi tensor or NumPy array, not {type(img)}"
            )

        self.core.set_img(self.img.ctypes.data)

    def circle(self, pos, color=0xFFFFFF, radius=1):
        self.canvas.circle_single(pos[0], pos[1], color, radius)

    def circles(self, pos, color=0xFFFFFF, radius=1):
        n = pos.shape[0]
        if len(pos.shape) == 3:
            assert pos.shape[2] == 1
            pos = pos[:, :, 0]

        assert pos.shape == (n, 2)
        pos = np.ascontiguousarray(pos.astype(np.float32))
        # Note: do not use "pos = int(pos.ctypes.data)" here
        # Otherwise pos will get garbage collected by Python
        # and the pointer to its data becomes invalid
        pos_ptr = int(pos.ctypes.data)

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

        self.canvas.circles_batched(n, pos_ptr, color_single, color_array,
                                    radius_single, radius_array)

    def triangle(self, a, b, c, color=0xFFFFFF):
        self.canvas.triangle_single(a[0], a[1], b[0], b[1], c[0], c[1], color)

    def line(self, begin, end, radius=1, color=0xFFFFFF):
        self.canvas.path_single(begin[0], begin[1], end[0], end[1], color,
                                radius)

    def rect(self, topleft, bottomright, radius=1, color=0xFFFFFF):
        a = topleft[0], topleft[1]
        b = bottomright[0], topleft[1]
        c = bottomright[0], bottomright[1]
        d = topleft[0], bottomright[1]
        self.line(a, b, radius, color)
        self.line(b, c, radius, color)
        self.line(c, d, radius, color)
        self.line(d, a, radius, color)

    def text(self, content, pos, font_size=15, color=0xFFFFFF):
        import taichi as ti
        # TODO: refactor Canvas::text
        font_size = float(font_size)
        pos = ti.vec(*pos)
        r, g, b = (color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff
        color = ti.vec(r / 255, g / 255, b / 255, 1)
        self.canvas.text(content, pos, font_size, color)

    def show(self, file=None):
        self.core.update()
        if file:
            self.core.screenshot(file)
        self.clear()

    class EventFilter:
        def __init__(self, *filter):
            self.filter = set()
            for ent in filter:
                if isinstance(ent, (list, tuple)):
                    type, key = ent
                    ent = (type, key)
                self.filter.add(ent)

        def match(self, e):
            if (e.type, e.key) in self.filter:
                return True
            if e.type in self.filter:
                return True
            if e.key in self.filter:
                return True
            return False

    def has_key_event(self):
        return self.core.has_key_event()

    def get_event(self, *filter):
        for e in self.get_events(*filter):
            self.event = e
            return True
        else:
            return False

    def get_events(self, *filter):
        filter = filter and GUI.EventFilter(*filter) or None

        while True:
            if not self.has_key_event():
                break
            e = self.get_key_event()
            if filter is None or filter.match(e):
                yield e

    def get_key_event(self):
        self.core.wait_key_event()
        e = GUI.Event()
        e.key = self.core.get_key_event_head_key()
        e.type = self.core.get_key_event_head_type()
        e.pos = self.core.get_key_event_head_pos()
        e.pos = (e.pos[0], e.pos[1])
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
            if key in self.key_pressed:
                return True
        else:
            return False

    def get_cursor_pos(self):
        pos = self.core.get_cursor_pos()
        return pos[0], pos[1]

    def has_key_pressed(self):
        import warnings
        warnings.warn(
            'gui.has_key_pressed() is deprecated, use gui.get_event() instead.',
            DeprecationWarning,
            stacklevel=3)
        if self.has_key_event():
            self.get_key_event()  # pop to update self.key_pressed
        return len(self.key_pressed) != 0

    @property
    def running(self):
        return not self.core.should_close

    @running.setter
    def running(self, value):
        if value:
            self.core.should_close = 0
        elif not self.core.should_close:
            self.core.should_close = 1


def rgb_to_hex(c):
    to255 = lambda x: min(255, max(0, int(x * 255)))
    return 65536 * to255(c[0]) + 256 * to255(c[1]) + to255(c[2])
