import datetime
import functools
import json
import os

import jsbeautifier


def get_benchmark_dir():
    return os.path.dirname(os.path.realpath(__file__))


def dump2json(obj):
    obj2dict = obj if type(obj) is dict else obj.__dict__
    options = jsbeautifier.default_options()
    options.indent_size = 4
    return jsbeautifier.beautify(json.dumps(obj2dict), options)


def datatime_with_format():
    return datetime.datetime.now().isoformat()
