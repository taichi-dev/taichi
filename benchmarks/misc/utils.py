import taichi as ti

kibibyte = 1024

bls2str = {False: "BLS_off", True: "BLS_on"}
dense2str = {False: "Struct_for", True: "Range_for"}

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


def geometric_mean(data_array):
    product = 1
    for data in data_array:
        product *= data
    return pow(product, 1.0 / len(data_array))


def md_table_header(suite_name, arch, test_dsize, test_repeat,
                    results_evaluation):
    header = '|' + suite_name + '.' + ti.core.arch_name(arch) + '|'
    header += ''.join('|' for i in range(len(test_dsize)))
    header += ''.join(item.__name__ + '|' for item in results_evaluation)

    layout = '|:--:|'
    layout += ''.join(
        ':--:|' for i in range(len(test_dsize) + len(results_evaluation)))

    size = '|**data size**|'
    size += ''.join(size2str(size) + '|' for size in test_dsize)
    size += ''.join('|' for i in range(len(results_evaluation)))

    repeat = '|**repeat**|'
    repeat += ''.join(
        str(scale_repeat(arch, size, test_repeat)) + '|'
        for size in test_dsize)
    repeat += ''.join('|' for i in range(len(results_evaluation)))

    lines = [header, layout, size, repeat]
    return lines
