import sys
import os
import shutil
import platform
import random
def get_os_name():
    name = platform.platform()
    if name.lower().startswith('darwin'):
        return 'osx'
    elif name.lower().startswith('windows'):
        return 'win'
    elif name.lower().startswith('linux'):
        return 'linux'
    assert False, "Unknown platform name %s" % name

if get_os_name() == 'osx':
    if os.path.exists('libtaichi_dll.dylib'):
        shutil.copy('libtaichi_dll.dylib', 'taichi.so')
        sys.path.append(".")
    import taichi as tc
elif get_os_name() == 'linux':
    if os.path.exists('libtaichi_dll.so'):
        shutil.copy('libtaichi_dll.so', 'taichi.so')
        sys.path.append(".")
    import taichi as tc
elif get_os_name() == 'win':
    dll_path = 'Release/taichi_dll.dll'
    d = 'tmp' + str(random.randint(0, 100000000)) + '/'
    try:
        os.mkdir(d)
    except:
        pass

    if os.path.exists(dll_path):
        shutil.copy(dll_path, d + 'taichi.pyd')
        sys.path.append(os.getcwd() + '/' + d)
        print sys.path
        import taichi as tc
    else:
        assert False, "libtaichi doesn't exists."
print sys.path

print "*Library Taichi Loaded.*"
import copy
import pyglet
import numpy as np
import ctypes

TEXTURE_PATH = '../assets/textures/'

def config_from_dict(args):
    d = copy.deepcopy(args)
    for k in d:
        d[k] = str(d[k])
    return tc.config_from_dict(d)

def make_polygon(points, scale):
    polygon = tc.Vector2List()
    for p in points:
        if type(p) == list or type(p) == tuple:
            polygon.append(scale * Vector(p[0], p[1]))
        else:
            polygon.append(scale * p)
    return polygon


def Vector(*args):
    if isinstance(args[0], tuple):
        args = tuple(*args)
    if len(args) == 2:
        v = tc.Vector2()
        v.x = float(args[0])
        v.y = float(args[1])
        return v
    elif len(args) == 3:
        v = tc.Vector3()
        v.x = float(args[0])
        v.y = float(args[1])
        v.z = float(args[2])
        return v
    else:
        assert False


def default_const_or_evaluate(f, default, u, v):
    if f == None:
        return default
    if type(f) in [float, int, tuple]:
        return f
    return f(u, v)

def const_or_evaluate(f, u, v):
    if type(f) in [float, int, tuple, tc.Vector2, tc.Vector3]:
        return f
    return f(u, v)


def array2d_to_image(arr, width, height, color_255, transform='levelset'):
    rasterized = arr.rasterize(width, height)
    raw_data = np.empty((width * height,), dtype='float32')
    rasterized.to_ndarray(raw_data.ctypes.data_as(ctypes.c_void_p).value)
    if transform == 'levelset':
        raw_data = (raw_data <= 0)
    else:
        x0, x1 = transform
        raw_data = (np.clip(raw_data, x0, x1) - x0) / (x1 - x0)
    dat = np.outer(np.ones_like(raw_data), color_255).astype('uint8')
    dat.reshape((len(raw_data), 4))
    dat[:, 3] = (color_255[3] * raw_data).astype('uint8')
    image_data = pyglet.image.ImageData(width, height, 'RGBA', dat.tostring())
    return image_data

def image_buffer_to_image(arr):
    raw_data = np.empty((arr.get_width() * arr.get_height() * 3,), dtype='float32')
    arr.to_ndarray(raw_data.ctypes.data_as(ctypes.c_void_p).value)
    dat = (raw_data * 255.0).astype('uint8')
    dat.reshape((len(raw_data) / 3, 3))
    data_string = dat.tostring()
    image_data = pyglet.image.ImageData(arr.get_width(), arr.get_height(), 'RGB', data_string)
    return image_data

def arange(x, y, d):
    while x < y:
        yield x
        x += d

