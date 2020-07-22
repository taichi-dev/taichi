from .core import taichi_lang_core as ti_core


class ActionRecord:
    def __init__(self, output):
        self.output = output

    def __enter__(self):
        ti_core.start_recording(self.output)

    def __exit__(self, type, value, tb):
        ti_core.stop_recording()
