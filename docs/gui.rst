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
        ``img`` entries will be clipped into range ``[0, 1]``.


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

  ti.GUI.ESCAPE
  ti.GUI.SHIFT
  ti.GUI.LEFT
  'a'
  'b'
  ...
  ti.GUI.LMB
  ti.GUI.RMB

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

    ``ti.GUI.EXIT`` occurs when you click on the close button / X button of a widnow.
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
            elif e.type == ti.GUI.SPACE:
                do_something()
            elif e.type in ['a', ti.GUI.LEFT]:
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

.. code-block:: python

    img = ti.imread('hello.png')
    ti.imshow(img, 'Window Title')
    ti.imwrite(img, 'hello2.png')

TODO: complete here
