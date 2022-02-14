import taichi as ti


@ti.func
def taichi_logo(pos: ti.template(), scale: float = 1 / 1.11):
    p = (pos - 0.5) / scale + 0.5
    ret = -1
    if not (p - 0.50).norm_sqr() <= 0.52**2:
        if ret == -1:
            ret = 0
    if not (p - 0.50).norm_sqr() <= 0.495**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return 1 - ret
