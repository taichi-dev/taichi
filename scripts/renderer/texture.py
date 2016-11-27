from taichi_utils import *

class Texture:
    def __init__(self, name, **kwargs):
        self.c = tc.create_texture(name)
        for key in kwargs:
            if isinstance(kwargs[key], Texture):
                kwargs[key] = kwargs[key].id
        self.c.initialize(P(**kwargs))
        self.id = tc.register_texture(self.c)

    @staticmethod
    def wrap_texture(value):
        if isinstance(value, tuple):
            return Texture('const', value=value)
        elif isinstance(value, float) or isinstance(value, int):
            return Texture('const', value=(value, value, value))
        else:
            return value

    def __int__(self):
        return self.id

    def __mul__(self, other):
        other = self.wrap_texture(other)
        return Texture("mul", tex1=self, tex2=other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = self.wrap_texture(other)
        return Texture("linear_op", alpha=1, tex1=self, beta=1, tex2=other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self.wrap_texture(other)
        return Texture("linear_op", alpha=1, tex1=self, beta=-1, tex2=other)

    def __rsub__(self, other):
        other = self.wrap_texture(other)
        return Texture("linear_op", alpha=-1, tex1=self, beta=1, tex2=other)

    def clamp(self):
        return Texture("linear_op", alpha=1, tex1=self, beta=0, tex2=self, need_clamp=True)

    @staticmethod
    def create_taichi_wallpaper(n):
        taichi = Texture('taichi')
        rep = Texture("repeater", repeat_u=n, repeat_v=n, tex=taichi)
        rep = Texture("checkerboard", tex1=rep, tex2=0 * rep + 0.2, repeat_u=n, repeat_v=n)
        return rep.clamp()
