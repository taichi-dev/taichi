.. _gui:

GUI system
==========

Taichi has a built-in GUI system to help users visualize results.


Create a window
---------------


.. function:: ti.GUI(title, res, bgcolor = 0x000000)

    :parameter title: (string) the window title
    :parameter res: (scalar or tuple) resolution / size of the window
    :parameter bgcolor: (optional, RGB hex) background color of the window
    :return: (GUI) an object represents the window

    Create a window.
    If ``res`` is scalar, then width will be equal to height.

    The following code creates a window of resolution ``640x360``:

    ::

        gui = ti.GUI('Window Title', (640, 360))


.. function:: gui.show(filename = None)

    :parameter gui: (GUI) the window object
    :parameter filename: (optional, string) see notes below

    Show the window on the screen.

    .. note::
        If ``filename`` is specified, a screenshot will be saved to the file specified by the name.
        For example, the following saves frames of the window to ``.png``'s:

        ::

            for frame in range(10000):
                render(img)
                gui.set_image(img)
                gui.show(f'{frame:06d}.png')


Paint on a window
-----------------


.. function:: gui.set_image(img)

    :parameter gui: (GUI) the window object
    :parameter img: (np.array or Tensor) tensor containing the image, see notes below

    Set an image to display on the window.

    The image pixels are set from the values of ``img[i, j]``, where ``i`` indicates the horizontal
    coordinates (from left to right) and ``j`` the vertical coordinates (from bottom to top).


    If the window size is ``(x, y)``, then ``img`` must be one of:

    * ``ti.var(shape=(x, y))``, a grey-scale image

    * ``ti.var(shape=(x, y, 3))``, where `3` is for ``(r, g, b)`` channels

    * ``ti.Vector(3, shape=(x, y))`` (see :ref:`vector`)

    * ``np.ndarray(shape=(x, y))``

    * ``np.ndarray(shape=(x, y, 3))``


    The data type of ``img`` must be one of:

    * ``uint8``, range ``[0, 255]``

    * ``uint16``, range ``[0, 65535]``

    * ``uint32``, range ``[0, 4294967295]``

    * ``float32``, range ``[0, 1]``

    * ``float64``, range ``[0, 1]``

    .. note ::

        When using ``float32`` or ``float64`` as the data type,
        ``img`` entries will be clipped into range ``[0, 1]`` for display.


.. function:: gui.circle(pos, color = 0xFFFFFF, radius = 1)

    :parameter gui: (GUI) the window object
    :parameter pos: (tuple of 2) the position of the circle
    :parameter color: (optional, RGB hex) the color to fill the circle
    :parameter radius: (optional, scalar) the radius of the circle

    Draw a solid circle.


.. function:: gui.circles(pos, color = 0xFFFFFF, radius = 1)

    :parameter gui: (GUI) the window object
    :parameter pos: (np.array) the positions of the circles
    :parameter color: (optional, RGB hex or np.array of uint32) the color(s) to fill the circles
    :parameter radius: (optional, scalar or np.array of float32) the radius (radii) of the circles

    Draw solid circles.

.. note::

    If ``color`` is a numpy array, the circle at ``pos[i]`` will be colored with ``color[i]``.
    In this case, ``color`` must have the same size as ``pos``.


.. function:: gui.line(begin, end, color = 0xFFFFFF, radius = 1)

    :parameter gui: (GUI) the window object
    :parameter begin: (tuple of 2) the first end point position of line
    :parameter end: (tuple of 2) the second end point position of line
    :parameter color: (optional, RGB hex) the color of line
    :parameter radius: (optional, scalar) the width of line

    Draw a line.


.. function:: gui.triangle(a, b, c, color = 0xFFFFFF)

    :parameter gui: (GUI) the window object
    :parameter a: (tuple of 2) the first end point position of triangle
    :parameter b: (tuple of 2) the second end point position of triangle
    :parameter c: (tuple of 2) the third end point position of triangle
    :parameter color: (optional, RGB hex) the color to fill the triangle

    Draw a solid triangle.


.. function:: gui.rect(topleft, bottomright, radius = 1, color = 0xFFFFFF)

    :parameter gui: (GUI) the window object
    :parameter topleft: (tuple of 2) the top-left point position of rectangle
    :parameter bottomright: (tuple of 2) the bottom-right point position of rectangle
    :parameter color: (optional, RGB hex) the color of stroke line
    :parameter radius: (optional, scalar) the width of stroke line

    Draw a hollow rectangle.


.. function:: gui.text(content, pos, font_size = 15, color = 0xFFFFFF)

    :parameter gui: (GUI) the window object
    :parameter content: (str) the text to draw
    :parameter pos: (tuple of 2) the top-left point position of the fonts / texts
    :parameter font_size: (optional, scalar) the size of font (in height)
    :parameter color: (optional, RGB hex) the foreground color of text

    Draw a line of text on screen.


Event processing
----------------

Every event have a key and type.
*Event key* is the key that you pressed on keyboard or mouse, can be one of:

::

  ti.GUI.ESCAPE  # Esc
  ti.GUI.SHIFT   # Shift
  ti.GUI.LEFT    # Left Arrow
  'a'            # we use lowercase for alphabet
  'b'
  ...
  ti.GUI.LMB     # Left Mouse Button
  ti.GUI.RMB     # Right Mouse Button

*Event type* is the type of event, for now, there are just three type of event:

::

  ti.GUI.RELEASE  # key up
  ti.GUI.PRESS    # key down
  ti.GUI.MOTION   # mouse moved


A *event filter* is a list combined of *key*, *type* and *(type, key)* tuple, e.g.:

.. code-block:: python

    # if ESC pressed or released:
    gui.get_event(ti.GUI.ESCAPE)

    # if any key is pressed:
    gui.get_event(ti.GUI.PRESS)

    # if ESC pressed or SPACE released:
    gui.get_event((ti.GUI.PRESS, ti.GUI.ESCAPE), (ti.GUI.RELEASE, ti.GUI.SPACE))


.. attribute:: gui.running

    :parameter gui: (GUI)
    :return: (bool) ``True`` if ``ti.GUI.EXIT`` event occurred, vice versa

    ``ti.GUI.EXIT`` occurs when you click on the close (X) button of a window.
    So ``gui.running`` will obtain ``False`` when the GUI is being closed.

    For example, loop until the close button is clicked:

    ::

        while gui.running:
            render()
            gui.set_image(pixels)
            gui.show()


    You can also close the window by manually setting ``gui.running`` to ``False``:

    ::

        while gui.running:
            if gui.get_event(ti.GUI.ESCAPE):
                gui.running = False

            render()
            gui.set_image(pixels)
            gui.show()


.. function:: gui.get_event(a, ...)

    :parameter gui: (GUI)
    :parameter a: (optional, EventFilter) filter out matched events
    :return: (bool) ``False`` if there is no pending event, vise versa

    Try to pop a event from the queue, and store it in ``gui.event``.

    For example:

    ::

        if gui.get_event():
            print('Got event, key =', gui.event.key)


    For example, loop until ESC is pressed:

    ::

        gui = ti.GUI('Title', (640, 480))
        while not gui.get_event(ti.GUI.ESCAPE):
            gui.set_image(img)
            gui.show()


.. function:: gui.get_events(a, ...)

    :parameter gui: (GUI)
    :parameter a: (optional, EventFilter) filter out matched events
    :return: (generator) a python generator, see below

    Basically the same as ``gui.get_event``, except for this one returns a generator of events instead of storing into ``gui.event``:

    ::

        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                do_something()
            elif e.key in ['a', ti.GUI.LEFT]:
                ...


.. function:: gui.is_pressed(key, ...)

    :parameter gui: (GUI)
    :parameter key: (EventKey) keys you want to detect
    :return: (bool) ``True`` if one of the keys pressed, vice versa

    .. warning::

        Must be used together with ``gui.get_event``, or it won't be updated!
        For example:

        ::

            while True:
                gui.get_event()  # must be called before is_pressed
                if gui.is_pressed('a', ti.GUI.LEFT):
                    print('Go left!')
                elif gui.is_pressed('d', ti.GUI.RIGHT):
                    print('Go right!')


.. function:: gui.get_cursor_pos()

    :parameter gui: (GUI)
    :return: (tuple of 2) current cursor position within the window

    For example:

    ::

        mouse_x, mouse_y = gui.get_cursor_pos()


Image I/O
---------

.. function:: ti.imwrite(img, filename)

    :parameter img: (Matrix or Expr) the image you want to export
    :parameter filename: (string) the location you want to save to

    Export a ``np.ndarray`` or Taichi tensor (``ti.Matrix``, ``ti.Vector``, or ``ti.var``) to a specified location ``filename``.

    Same as ``ti.GUI.show(filename)``, the format of the exported image is determined by **the suffix of** ``filename`` as well. Now ``ti.imwrite`` supports exporting images to ``png``, ``img`` and ``jpg`` and we recommend using ``png``.

    Please make sure that the input image has **a valid shape**. If you want to export a grayscale image, the input shape of tensor should be ``(height, weight)`` or ``(height, weight, 1)``. For example:

    .. code-block:: python

        import taichi as ti

        ti.init()

        shape = (512, 512)
        type = ti.u8
        pixels = ti.var(dt=type, shape=shape)

        @ti.kernel
        def draw():
            for i, j in pixels:
                pixels[i, j] = ti.random() * 255    # integars between [0, 255] for ti.u8

        draw()

        ti.imwrite(pixels, f"export_u8.png")

    Besides, for RGB or RGBA images, ``ti.imwrite`` needs to receive a tensor which has shape ``(height, width, 3)`` and ``(height, width, 4)`` individually.

    Generally the value of the pixels on each channel of a ``png`` image is an integar in [0, 255]. For this reason, ``ti.imwrite`` will **cast tensors** which has different datatypes all **into integars between [0, 255]**. As a result, ``ti.imwrite`` has the following requirements for different datatypes of input tensors:

    - For float-type (``ti.f16``, ``ti.f32``, etc) input tensors, **the value of each pixel should be float between [0.0, 1.0]**. Otherwise ``ti.imwrite`` will first clip them into [0.0, 1.0]. Then they are multiplied by 256 and casted to integaters ranging from [0, 255].

    - For int-type (``ti.u8``, ``ti.u16``, etc) input tensors, **the value of each pixel can be any valid integer in its own bounds**. These integers in this tensor will be scaled to [0, 255] by being divided over the upper bound of its basic type accordingly.

    Here is another example:

    .. code-block:: python

        import taichi as ti

        ti.init()

        shape = (512, 512)
        channels = 3
        type = ti.f32
        pixels = ti.Matrix(channels, dt=type, shape=shape)

        @ti.kernel
        def draw():
            for i, j in pixels:
                for k in ti.static(range(channels)):
                    pixels[i, j][k] = ti.random()   # floats between [0, 1] for ti.f32

        draw()

        ti.imwrite(pixels, f"export_f32.png")


.. function:: ti.imread(filename, channels=0)

    :parameter filename: (string) the filename of the image to load
    :parameter channels: (optional int) the number of channels in your specified image. The default value ``0`` means the channels of the returned image is adaptive to the image file

    :return: (np.ndarray) the image read from ``filename``

    This function loads an image from the target filename and returns it as a ``np.ndarray(dtype=np.uint8)``.

    Each value in this returned tensor is an integer in [0, 255].

.. function:: ti.imshow(img, windname)

    :parameter img: (Matrix or Expr) the image to show in the GUI
    :parameter windname: (string) the name of the GUI window

    This function will create an instance of ``ti.GUI`` and show the input image on the screen.

    It has the same logic as ``ti.imwrite`` for different datatypes.
