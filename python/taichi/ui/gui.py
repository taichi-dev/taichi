import pathlib

from taichi.core import ti_core as _ti_core
from taichi.lang.impl import default_cfg
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.ops import get_addr

from .utils import *


class Gui:
    def __init__(self, gui) -> None:
        self.gui = gui  #reference to a PyGui

    def begin(self, name, x, y, width, height):
        self.gui.begin(name, x, y, width, height)

    def end(self):
        self.gui.end()

    def text(self, text):
        self.gui.text(text)

    def checkbox(self, text, old_value):
        return self.gui.checkbox(text, old_value)

    def slider_float(self, text, old_value, minimum, maximum):
        return self.gui.slider_float(text, old_value, minimum, maximum)

    def color_edit_3(self, text, old_value):
        return self.gui.color_edit_3(text, old_value)

    def button(self, text):
        return self.gui.button(text)
