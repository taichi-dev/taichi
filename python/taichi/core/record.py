import os

from taichi.core import util


def record_action_entry(name, contents):
    util.ti_core.record_action_entry(name, list(contents.items()))


def record_action_hint(name, content=None):
    if content is None:
        name, content = 'hint', name
    record_action_entry(name, {'content': content})


def record_action_config(key, value):
    record_action_entry('config', {'key': key, 'value': value})


def start_recording(filename):
    util.ti_core.start_recording(filename)


def stop_recording():
    util.ti_core.stop_recording()


class RecordKernelGroup:
    def __init__(self, name):
        if name in RecordKernelGroup.recorded:
            self.name = None
        else:
            RecordKernelGroup.recorded.add(name)
            self.name = name

    def __enter__(self):
        if self.name is not None:
            record_action_hint('group_begin', self.name)
        return self

    def __exit__(self, *args):
        if self.name is not None:
            record_action_hint('group_end', self.name)

    recorded = set()


record_file = os.environ.get('TI_ACTION_RECORD')
if record_file:
    start_recording(record_file)

__all__ = [
    'start_recording',
    'stop_recording',
    'record_action_hint',
    'record_action_entry',
    'record_action_config',
    'RecordKernelGroup',
]
