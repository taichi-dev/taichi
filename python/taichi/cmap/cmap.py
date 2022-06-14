import taichi as ti
from taichi.math import cos, mix, pi, sin, smoothstep, step, vec2, vec3


@ti.func
def _cubehelix(c):
    sc = vec2(sin(c.x), cos(c.x))
    return c.z * (1.0 + c.y * (1.0 - c.z) *
                  (sc.x * vec3(0.14861, 0.29227, -1.97294) +
                   sc.y * vec3(1.78277, -0.90649, 0.0)))


@ti.func
def rainbow(x: float):
    return _cubehelix(
        vec3(2 * pi * x - 1.74533,
             (0.25 * cos(2 * pi * x) + 0.25) * vec2(-1.5, -0.9) +
             vec2(1.5, 0.8)))


@ti.func
def hot(x: float):
    return vec3(smoothstep(0.00, 0.33, x), smoothstep(0.33, 0.66, x),
                smoothstep(0.66, 1.00, x))


@ti.func
def _rescale(xmin, xmax, x):
    return (x - xmin) / (xmax - xmin)


@ti.func
def jet(x: float):
    color0 = vec3(0, 0, 0.5625)
    u1 = 1 / 9
    color1 = vec3(0.0, 0.0, 1.0)
    u2 = 23 / 63
    color2 = vec3(0.0, 1.0, 1.0)
    u3 = 13 / 21
    color3 = vec3(1.0, 1.0, 0.0)
    u4 = 47 / 63
    color4 = vec3(1.0, 0.5, 0.0)
    u5 = 55 / 63
    color5 = vec3(1.0, 0.0, 0.0)
    color6 = vec3(0.5, 0.0, 0.0)
    return (mix(color0, color1, _rescale(0.0, u1, x)) +
            (mix(color1, color2, _rescale(u1, u2, x)) -
             mix(color0, color1, _rescale(0., u1, x))) * step(u1, x) +
            (mix(color2, color3, _rescale(u2, u3, x)) -
             mix(color1, color2, _rescale(u1, u2, x))) * step(u2, x) +
            (mix(color3, color4, _rescale(u3, u4, x)) -
             mix(color2, color3, _rescale(u2, u3, x))) * step(u3, x) +
            (mix(color4, color5, _rescale(u4, u5, x)) -
             mix(color3, color4, _rescale(u3, u4, x))) * step(u4, x) +
            (mix(color5, color6, _rescale(u5, 1., x)) -
             mix(color4, color5, _rescale(u4, u5, x))) * step(u5, x))


@ti.func
def plasma(x: float):
    c0 = vec3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754)
    c1 = vec3(2.176514634195958, 0.2383834171260182, 0.7539604599784036)
    c2 = vec3(-2.689460476458034, -7.455851135738909, 3.110799939717086)
    c3 = vec3(6.130348345893603, 42.3461881477227, -28.51885465332158)
    c4 = vec3(-11.10743619062271, -82.66631109428045, 60.13984767418263)
    c5 = vec3(10.02306557647065, 71.41361770095349, -54.07218655560067)
    c6 = vec3(-3.658713842777788, -22.93153465461149, 18.19190778539828)
    return c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6)))))


@ti.func
def viridis(x: float):
    c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061)
    c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685)
    c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659)
    c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987)
    c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105)
    c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234)
    c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832)
    return c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6)))))


__all__ = ["hot", "jet", "plasma", "rainbow", "viridis"]
