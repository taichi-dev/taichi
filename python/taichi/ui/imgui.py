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
        """Declares an integer slider, and returns its newest value.

        Args:
            text (str): a line of text to be shown next to the slider.
            old_value (int or tuple): the current value. Can be a scalar int
                or a tuple of 2-4 ints for multi-component sliders.
            minimum (int): the minimum value of the slider.
            maximum (int): the maximum value of the slider.

        Returns:
            int or tuple: the updated value (same type as input).
        """
        if isinstance(old_value, (tuple, list)):
            n = len(old_value)
            if n == 2:
                return self.gui.slider_int2(text, tuple(old_value), minimum, maximum)
            if n == 3:
                return self.gui.slider_int3(text, tuple(old_value), minimum, maximum)
            if n == 4:
                return self.gui.slider_int4(text, tuple(old_value), minimum, maximum)
            raise ValueError(f"slider_int expects 1-4 components, got {n}")
        return self.gui.slider_int(text, old_value, minimum, maximum)

    def slider_float(self, text, old_value, minimum, maximum):
        """Declares a float slider, and returns its newest value.

        Args:
            text (str): a line of text to be shown next to the slider.
            old_value (float or tuple): the current value. Can be a scalar float
                or a tuple of 2-4 floats for multi-component sliders.
            minimum (float): the minimum value of the slider.
            maximum (float): the maximum value of the slider.

        Returns:
            float or tuple: the updated value (same type as input).
        """
        if isinstance(old_value, (tuple, list)):
            n = len(old_value)
            if n == 2:
                return self.gui.slider_float2(text, tuple(old_value), minimum, maximum)
            if n == 3:
                return self.gui.slider_float3(text, tuple(old_value), minimum, maximum)
            if n == 4:
                return self.gui.slider_float4(text, tuple(old_value), minimum, maximum)
            raise ValueError(f"slider_float expects 1-4 components, got {n}")
        return self.gui.slider_float(text, old_value, minimum, maximum)

    def color_edit(self, text, old_value):
        """Declares a color edit palette.

        Auto-detects RGB vs RGBA based on tuple size.

        Args:
            text (str): a line of text to be shown next to the palette.
            old_value (tuple): the current color value. Can be a tuple of
                3 floats (RGB) or 4 floats (RGBA) in [0,1].

        Returns:
            tuple: the updated color (same size as input).
        """
        n = len(old_value)
        if n == 3:
            return self.gui.color_edit_3(text, tuple(old_value))
        if n == 4:
            return self.gui.color_edit_4(text, tuple(old_value))
        raise ValueError(f"color_edit expects 3 (RGB) or 4 (RGBA) components, got {n}")

    def color_edit_3(self, text, old_value):
        """Declares an RGB color edit palette.

        Note:
            Prefer using :meth:`color_edit` which auto-detects RGB/RGBA.

        Args:
            text (str): a line of text to be shown next to the palette.
            old_value (Tuple[float]): the current value of the color, this \
                should be a tuple of 3 floats in [0,1] that indicates RGB values.

        Returns:
            tuple: the updated RGB color as (r, g, b).
        """
        return self.gui.color_edit_3(text, old_value)

    def color_picker(self, text, old_value):
        """Declares a full color picker widget with color wheel/square.

        Auto-detects RGB vs RGBA based on tuple size.

        Args:
            text (str): a line of text to be shown next to the picker.
            old_value (tuple): the current color value. Can be a tuple of
                3 floats (RGB) or 4 floats (RGBA) in [0,1].

        Returns:
            tuple: the updated color (same size as input).
        """
        n = len(old_value)
        if n == 3:
            return self.gui.color_picker_3(text, tuple(old_value))
        if n == 4:
            return self.gui.color_picker_4(text, tuple(old_value))
        raise ValueError(f"color_picker expects 3 (RGB) or 4 (RGBA) components, got {n}")

    def button(self, text):
        """Declares a button, and returns whether or not it had just been clicked.

        Args:
            text (str): a line of text to be shown next to the button.
        """
        return self.gui.button(text)

    def input_int(self, label, old_value):
        """Integer input field with +/- buttons.

        Args:
            label (str): Label for the input field.
            old_value (int or tuple): Current value. Can be a scalar int
                or a tuple of 2-4 ints for multi-component inputs.

        Returns:
            int or tuple: The updated value (same type as input).
        """
        if isinstance(old_value, (tuple, list)):
            n = len(old_value)
            if n == 2:
                return self.gui.input_int2(label, tuple(old_value))
            if n == 3:
                return self.gui.input_int3(label, tuple(old_value))
            if n == 4:
                return self.gui.input_int4(label, tuple(old_value))
            raise ValueError(f"input_int expects 1-4 components, got {n}")
        return self.gui.input_int(label, old_value)

    def input_float(self, label, old_value):
        """Float input field with +/- buttons.

        Args:
            label (str): Label for the input field.
            old_value (float or tuple): Current value. Can be a scalar float
                or a tuple of 2-4 floats for multi-component inputs.

        Returns:
            float or tuple: The updated value (same type as input).
        """
        if isinstance(old_value, (tuple, list)):
            n = len(old_value)
            if n == 2:
                return self.gui.input_float2(label, tuple(old_value))
            if n == 3:
                return self.gui.input_float3(label, tuple(old_value))
            if n == 4:
                return self.gui.input_float4(label, tuple(old_value))
            raise ValueError(f"input_float expects 1-4 components, got {n}")
        return self.gui.input_float(label, old_value)

    def drag_int(self, label, old_value, speed=1.0, minimum=0, maximum=0):
        """Draggable integer input.

        Args:
            label (str): Label for the input.
            old_value (int or tuple): Current value. Can be a scalar int
                or a tuple of 2-4 ints for multi-component inputs.
            speed (float): Drag speed multiplier.
            minimum (int): Minimum value (0 for no limit).
            maximum (int): Maximum value (0 for no limit).

        Returns:
            int or tuple: The updated value (same type as input).
        """
        if isinstance(old_value, (tuple, list)):
            n = len(old_value)
            if n == 2:
                return self.gui.drag_int2(label, tuple(old_value), speed, minimum, maximum)
            if n == 3:
                return self.gui.drag_int3(label, tuple(old_value), speed, minimum, maximum)
            if n == 4:
                return self.gui.drag_int4(label, tuple(old_value), speed, minimum, maximum)
            raise ValueError(f"drag_int expects 1-4 components, got {n}")
        return self.gui.drag_int(label, old_value, speed, minimum, maximum)

    def drag_float(self, label, old_value, speed=1.0, minimum=0.0, maximum=0.0):
        """Draggable float input.

        Args:
            label (str): Label for the input.
            old_value (float or tuple): Current value. Can be a scalar float
                or a tuple of 2-4 floats for multi-component inputs.
            speed (float): Drag speed multiplier.
            minimum (float): Minimum value (0.0 for no limit).
            maximum (float): Maximum value (0.0 for no limit).

        Returns:
            float or tuple: The updated value (same type as input).
        """
        if isinstance(old_value, (tuple, list)):
            n = len(old_value)
            if n == 2:
                return self.gui.drag_float2(label, tuple(old_value), speed, minimum, maximum)
            if n == 3:
                return self.gui.drag_float3(label, tuple(old_value), speed, minimum, maximum)
            if n == 4:
                return self.gui.drag_float4(label, tuple(old_value), speed, minimum, maximum)
            raise ValueError(f"drag_float expects 1-4 components, got {n}")
        return self.gui.drag_float(label, old_value, speed, minimum, maximum)

    def combo(self, label, current_index, items):
        """Combo box (dropdown) for selecting from a tuple of items.

        Args:
            label (str): Label for the combo box.
            current_index (int): Currently selected index (0-based).
            items (tuple[str, ...]): Tuple of string options.

        Returns:
            int: The newly selected index.

        Note:
            Converting Python strings to ImGui format is cached when the same
            tuple object is passed across frames. For best performance, define
            options at module or class level (not inside the render loop)::

                DIFFICULTIES = ("Easy", "Medium", "Hard")  # module level

                while window.running:
                    with gui.sub_window(...) as w:
                        # Cached - same tuple object each frame
                        idx = w.combo("Difficulty", idx, DIFFICULTIES)
                    window.show()
        """
        return self.gui.combo(label, current_index, items)

    def tree_node_push(self, label):
        """Begin a collapsible tree node (low-level).

        Args:
            label (str): Label for the tree node.

        Returns:
            bool: True if the node is expanded, False if collapsed.

        Note:
            You must call tree_node_pop() if this returns True.
            Consider using tree_node() generator instead for automatic cleanup.
        """
        return self.gui.tree_node_push(label)

    def tree_node_pop(self):
        """End a tree node (low-level).

        Only call this if tree_node_push() returned True.
        """
        self.gui.tree_node_pop()

    def tree_node(self, label):
        """Collapsible tree node (generator).

        Use with a for loop - the body runs only if the node is expanded,
        and tree_node_pop is called automatically.

        Args:
            label (str): Label for the tree node.

        Yields:
            Nothing, but loop body executes only if node is expanded.

        Example::

            for _ in gui.tree_node("Settings"):
                gui.slider_float("Volume", volume, 0, 1)
                for _ in gui.tree_node("Advanced"):
                    gui.checkbox("Debug", debug)
        """
        if self.gui.tree_node_push(label):
            try:
                yield
            finally:
                self.gui.tree_node_pop()

    def separator(self):
        """Draw a horizontal separator line."""
        self.gui.separator()

    def same_line(self):
        """Place the next widget on the same line as the previous one."""
        self.gui.same_line()

    @contextmanager
    def indent(self):
        """Indent subsequent widgets (context manager).

        Example::

            gui.text("Parent")
            with gui.indent():
                gui.text("Child 1")
                gui.text("Child 2")
            gui.text("Back to normal")
        """
        self.gui.indent()
        try:
            yield
        finally:
            self.gui.unindent()

    def progress_bar(self, fraction):
        """Display a progress bar.

        Args:
            fraction (float): Progress value between 0.0 and 1.0.
        """
        self.gui.progress_bar(fraction)

    def collapsing_header(self, label):
        """A collapsible header that reveals content when expanded.

        Args:
            label (str): Label for the header.

        Returns:
            bool: True if the header is expanded, False if collapsed.

        Example::

            if gui.collapsing_header("Advanced Settings"):
                gui.slider_float("Detail", detail, 0, 1)
        """
        return self.gui.collapsing_header(label)

    def selectable(self, label, selected=False):
        """A selectable item (like a list entry).

        Args:
            label (str): Label for the item.
            selected (bool): Whether the item is currently selected.

        Returns:
            bool: The new selected state (toggled if clicked).
        """
        return self.gui.selectable(label, selected)

    def radio_button(self, label, active):
        """A radio button.

        Args:
            label (str): Label for the button.
            active (bool): Whether this option is currently selected.

        Returns:
            bool: True if clicked (use to update selection state).

        Example::

            if gui.radio_button("Option A", choice == 0):
                choice = 0
            if gui.radio_button("Option B", choice == 1):
                choice = 1
        """
        return self.gui.radio_button(label, active)

    def listbox(self, label, current_index, items, height_in_items=-1):
        """A listbox for selecting from a tuple of items.

        Args:
            label (str): Label for the listbox.
            current_index (int): Currently selected index (0-based).
            items (tuple[str, ...]): Tuple of string options.
            height_in_items (int): Number of items to show (-1 for default).

        Returns:
            int: The newly selected index.

        Note:
            Like combo(), uses caching for tuple→string conversion.
            Define items as module-level constants for best performance.
        """
        return self.gui.listbox(label, current_index, items, height_in_items)

    def begin_tab_bar(self, bar_id):
        """Begin a tab bar (low-level).

        Args:
            bar_id (str): Unique identifier for the tab bar.

        Returns:
            bool: True if the tab bar is visible.

        Note:
            Must call end_tab_bar() if this returns True.
            Consider using tab_bar() generator instead.
        """
        return self.gui.begin_tab_bar(bar_id)

    def end_tab_bar(self):
        """End a tab bar (low-level).

        Only call this if begin_tab_bar() returned True.
        """
        self.gui.end_tab_bar()

    def begin_tab_item(self, label):
        """Begin a tab item (low-level).

        Args:
            label (str): Label for the tab.

        Returns:
            bool: True if this tab is selected.

        Note:
            Must call end_tab_item() if this returns True.
            Consider using tab_item() generator instead.
        """
        return self.gui.begin_tab_item(label)

    def end_tab_item(self):
        """End a tab item (low-level).

        Only call this if begin_tab_item() returned True.
        """
        self.gui.end_tab_item()

    def tab_bar(self, bar_id):
        """Tab bar container (generator).

        Use with a for loop - the body runs only if the tab bar is visible,
        and end_tab_bar is called automatically.

        Args:
            bar_id (str): Unique identifier for the tab bar.

        Yields:
            Nothing, but loop body executes only if visible.

        Example::

            for _ in gui.tab_bar("my_tabs"):
                for _ in gui.tab_item("Tab 1"):
                    gui.text("Content for tab 1")
                for _ in gui.tab_item("Tab 2"):
                    gui.text("Content for tab 2")
        """
        if self.gui.begin_tab_bar(bar_id):
            try:
                yield
            finally:
                self.gui.end_tab_bar()

    def tab_item(self, label):
        """Tab item (generator).

        Use with a for loop inside tab_bar - the body runs only if this tab
        is selected, and end_tab_item is called automatically.

        Args:
            label (str): Label for the tab.

        Yields:
            Nothing, but loop body executes only if tab is selected.

        Example::

            for _ in gui.tab_bar("settings"):
                for _ in gui.tab_item("General"):
                    gui.checkbox("Enable feature", enabled)
                for _ in gui.tab_item("Advanced"):
                    gui.slider_int("Level", level, 1, 10)
        """
        if self.gui.begin_tab_item(label):
            try:
                yield
            finally:
                self.gui.end_tab_item()

    def begin_table(self, table_id, columns):
        """Begin a table (low-level).

        Args:
            table_id (str): Unique identifier for the table.
            columns (int): Number of columns.

        Returns:
            bool: True if the table is visible and should be rendered.

        Note:
            Must call end_table() if this returns True.
            Consider using table() generator instead for automatic cleanup.
        """
        return self.gui.begin_table(table_id, columns)

    def end_table(self):
        """End a table (low-level).

        Only call this if begin_table() returned True.
        """
        self.gui.end_table()

    def table_setup_column(self, label):
        """Set up a column header for the table.

        Call after begin_table() and before the first row.

        Args:
            label (str): Label for the column header.
        """
        self.gui.table_setup_column(label)

    def table_headers_row(self):
        """Submit the header row after setting up columns.

        Call after all table_setup_column() calls and before data rows.
        """
        self.gui.table_headers_row()

    def table_next_row(self):
        """Begin a new row in the table."""
        self.gui.table_next_row()

    def table_next_column(self):
        """Move to the next column in the current row.

        Returns:
            bool: True if the column is visible (not clipped).
        """
        return self.gui.table_next_column()

    def table(self, table_id, columns):
        """Table container (generator).

        Use with a for loop - the body runs only if the table is visible,
        and end_table is called automatically.

        Args:
            table_id (str): Unique identifier for the table.
            columns (int): Number of columns.

        Yields:
            Nothing, but loop body executes only if table is visible.

        Example::

            for _ in gui.table("my_table", 3):
                gui.table_setup_column("ID")
                gui.table_setup_column("Name")
                gui.table_setup_column("Price")
                gui.table_headers_row()

                for item in items:
                    gui.table_next_row()
                    gui.table_next_column(); gui.text(str(item["id"]))
                    gui.table_next_column(); gui.text(item["name"])
                    gui.table_next_column(); gui.text(f"${item['price']:.2f}")
        """
        if self.gui.begin_table(table_id, columns):
            try:
                yield
            finally:
                self.gui.end_table()
