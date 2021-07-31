import taichi as ti

dtype2str = {ti.i32: "i32", ti.i64: "i64", ti.f32: "f32", ti.f64: "f64"}
dtype_size = {ti.i32: 4, ti.i64: 8, ti.f32: 4, ti.f64: 8}

# for output string
size_subsection = [(0.0, 'B'), (1024.0, 'KB'), (1048576.0, 'MB'),
                   (1073741824.0, 'GB'), (float('inf'), 'INF')]  #B KB MB GB


def size2str(size_in_byte):
    for dsize, units in reversed(size_subsection):
        if size_in_byte >= dsize:
            return str(round(size_in_byte / dsize, 4)) + units


def scale_repeat(arch, datasize, repeat=10):
    scaled = repeat
    if (arch == ti.gpu) | (arch == ti.opengl) | (arch == ti.cuda):
        scaled *= 10
    if datasize <= 4 * 1024 * 1024:
        scaled *= 10
    return scaled
