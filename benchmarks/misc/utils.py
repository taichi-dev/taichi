import datetime
import json

import jsbeautifier
from taichi._lib import core as ti_core

import taichi as ti


def dtype2str(ti_dtype):
    type_str_dict = {
        ti.i32: "i32",
        ti.i64: "i64",
        ti.f32: "f32",
        ti.f64: "f64"
    }
    if ti_dtype not in type_str_dict:
        raise RuntimeError('Unsupported ti.dtype: ' + str(type(ti_dtype)))
    else:
        return type_str_dict[ti_dtype]


def dtype_size(ti_dtype):
    dtype_size_dict = {ti.i32: 4, ti.i64: 8, ti.f32: 4, ti.f64: 8}
    if ti_dtype not in dtype_size_dict:
        raise RuntimeError('Unsupported ti.dtype: ' + str(type(ti_dtype)))
    else:
        return dtype_size_dict[ti_dtype]


def arch_name(arch):
    return str(arch).replace('Arch.', '')


def get_commit_hash():
    return ti_core.get_commit_hash()


def datatime_with_format():
    return datetime.datetime.now().isoformat()


def dump2json(obj):
    if type(obj) is dict:
        obj2dict = obj
    else:
        obj2dict = obj.__dict__
    options = jsbeautifier.default_options()
    options.indent_size = 4
    return jsbeautifier.beautify(json.dumps(obj2dict), options)


def size2str(size_in_byte):
    # for output string
    size_subsection = [(0.0, 'B'), (1024.0, 'KB'), (1048576.0, 'MB'),
                       (1073741824.0, 'GB'),
                       (float('inf'), 'INF')]  #B KB MB GB
    for dsize, units in reversed(size_subsection):
        if size_in_byte >= dsize:
            return str(round(size_in_byte / dsize, 4)) + units


def geometric_mean(data_array):
    product = 1
    for data in data_array:
        product *= data
    return pow(product, 1.0 / len(data_array))


def scaled_repeat_times(arch, datasize, repeat=1):
    if (arch == ti.gpu) | (arch == ti.opengl) | (arch == ti.cuda):
        repeat *= 10
    if datasize <= 4 * 1024 * 1024:
        repeat *= 10
    return repeat


def md_table_header(suite_name, arch, test_dsize, test_repeat,
                    results_evaluation):
    header = '|' + suite_name + '.' + arch_name(arch) + '|'
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
        str(scaled_repeat_times(arch, size, test_repeat)) + '|'
        for size in test_dsize)
    repeat += ''.join('|' for i in range(len(results_evaluation)))

    lines = [header, layout, size, repeat]
    return lines
