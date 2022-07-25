import os

from taichi._lib import core as ti_python_core


def record_action_entry(name, contents):
    ti_python_core.record_action_entry(name, list(contents.items()))


def record_action_hint(name, content=None):
    if content is None:
        name, content = 'hint', name
    record_action_entry(name, {'content': content})


def record_action_config(key, value):
    record_action_entry('config', {'key': key, 'value': value})


def start_recording(filename):
    """Starts recording kernel information to a `yml` file.

    Args:
        filename (str): output `yml` file.

    Example::

        >>> ti.aot.start_recording('record.yml')
        >>> ti.init(arch=ti.cc)
        >>> loss = ti.field(float, (), needs_grad=True)
        >>> x = ti.field(float, 233, needs_grad=True)
        >>>
        >>> @ti.kernel
        >>> def compute_loss():
        >>>     for i in x:
        >>>         loss[None] += x[i]**2
        >>>
        >>> @ti.kernel
        >>> def do_some_works():
        >>>     for i in x:
        >>>         x[i] -= x.grad[i]
        >>>
        >>> with ti.ad.Tape(loss):
        >>>     compute_loss()
        >>> do_some_works()
    """
    ti_python_core.start_recording(filename)


def stop_recording():
    """Stops recording kernel information.

    This function should be called in pair with :func:`~ti.aot.start_recording`.
    """
    ti_python_core.stop_recording()


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
]
