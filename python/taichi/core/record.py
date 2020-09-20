from .util import ti_core
import os


def record_action_hint(name, content=None):
    if content is None:
        name, content = 'hint', name
    ti_core.record_action_hint(name, content)


def start_recording(filename):
    ti_core.start_recording(filename)


def stop_recording():
    ti_core.stop_recording()


class RecordAction:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        start_recording(self.filename)
        return self

    def __exit__(self, *args):
        stop_recording()


class RecordGroupHint:
    def __init__(self, name):
        if name in recorded:
            self.name = None
        else:
            recorded.add(name)
            self.name = name

    def __enter__(self):
        if self.name is not None:
            record_action_hint('group_begin', self.name)
        return self

    def __exit__(self, *args):
        if self.name is not None:
            record_action_hint('group_end', self.name)

    recorded = {}


record_file = os.environ.get('TI_ACTION_RECORD')
if record_file:
    start_recording(record_file)

__all__ = [
    'start_recording',
    'stop_recording',
    'record_action_hint',
    'RecordAction',
    'RecordGroupHint',
]
