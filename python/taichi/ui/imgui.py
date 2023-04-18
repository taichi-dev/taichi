from contextlib import contextmanager


class Gui:
    """For declaring IMGUI components in a :class:`taichi.ui.Window`
    created by the GGUI system.

    Args:
        gui: reference to a `PyGui`.
    """

    def __init__(self, gui) -> None:
        self.gui = gui

    @contextmanager
    def sub_window(self, name, x, y, width, height):
        """Creating a context manager for subwindow.

        Note:
            All args of this method should align with `begin`.

        Args:
            x (float): The x-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            y (float): The y-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            width (float): The width of the subwindow relative to the full window.
            height (float): The height of the subwindow relative to the full window.

        Example::

            >>> with gui.sub_window(name, x, y, width, height) as g:
            >>>     g.text("Hello, World!")
        """
        self.begin(name, x, y, width, height)
        try:
            yield self
        finally:
            self.end()

    def begin(self, name, x, y, width, height):
        """Creates a subwindow that holds imgui widgets.

        All widget function calls (e.g. `text`, `button`) after the `begin`
        and before the next `end` will describe the widgets within this subwindow.

        Args:
            x (float): The x-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            y (float): The y-coordinate (between 0 and 1) of the top-left \
                corner of the subwindow, relative to the full window.
            width (float): The width of the subwindow relative to the full window.
            height (float): The height of the subwindow relative to the full window.
        """
        self.gui.begin(name, x, y, width, height)

    def end(self):
        """End the description of the current subwindow."""
        self.gui.end()

    def text(self, text, color=None):
        """Declares a line of text."""
        if color is None:
            self.gui.text(text)
        else:
            self.gui.text_colored(text, color)

    def checkbox(self, text, old_value):
        """Declares a checkbox, and returns whether or not it has been checked.

        Args:
            text (str): a line of text to be shown next to the checkbox.
            old_value (bool): whether the checkbox is currently checked.
        """
        return self.gui.checkbox(text, old_value)

    def slider_int(self, text, old_value, minimum, maximum):
        """Declares a slider, and returns its newest value.

        Args:
            text (str): a line of text to be shown next to the slider
            old_value (int) : the current value of the slider.
            minimum (int): the minimum value of the slider.
            maximum (int): the maximum value of the slider.

        Returns:
            int: the updated value of the slider.
        """
        return self.gui.slider_int(text, old_value, minimum, maximum)

    def slider_float(self, text, old_value, minimum, maximum):
        """Declares a slider, and returns its newest value.

        Args:
            text (str): a line of text to be shown next to the slider
            old_value (float): the current value of the slider.
            minimum (float): the minimum value of the slider.
            maximum (float): the maximum value of the slider.
        """
        return self.gui.slider_float(text, old_value, minimum, maximum)

    def color_edit_3(self, text, old_value):
        """Declares a color edit palate.

        Args:
            text (str): a line of text to be shown next to the palate.
            old_value (Tuple[float]): the current value of the color, this \
                should be a tuple of floats in [0,1] that indicates RGB values.
        """
        return self.gui.color_edit_3(text, old_value)

    def button(self, text):
        """Declares a button, and returns whether or not it had just been clicked.

        Args:
            text (str): a line of text to be shown next to the button.
        """
        return self.gui.button(text)
