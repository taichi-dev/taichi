.. _gui:

GUI system
==========

Taichi has a built-in GUI system to help users display graphic results easier.


Create a window
---------------


.. function:: ti.GUI(title, res, bgcolor = 0x000000)

    :parameter title: (string) the window title
    :parameter res: (scalar or tuple) resolution / size of the window
    :parameter bgcolor: (optional, RGB hex) background color of the window
    :return: (GUI) an object represents the window

    Create a window.
    If ``res`` is scalar, then width will be equal to height.

    This creates a window whose width is 1024, height is 768:

    ::

        gui = ti.GUI('Window Title', (1024, 768))


.. function:: gui.show(filename = None)

    :parameter gui: (GUI) the window object
    :parameter filename: (optional, string) see notes below

    Show the window on the screen.

    .. note::
        If `filename` is specified, screenshot will be saved to the file specified by the name. For example, this screenshots each frame of the window, and save it in ``.png``'s:

        ::

            for frame in range(10000):
                render(img)
                gui.set_image(img)
                gui.show(f'{frame:06d}.png')


Paint a window
--------------


.. function:: gui.set_image(img)

    :parameter gui: (GUI) the window object
    :parameter img: (np.array or Tensor) tensor containing the image, see notes below

    Set a image to display on the window.

    The pixel, ``i`` from bottom to up, ``j`` from left to right, is set to the value of ``img[i, j]``.


    If the window size is ``(x, y)``, then the ``img`` must be one of:

    * ``ti.var(shape=(x, y))``, a grey-scale image

    * ``ti.var(shape=(x, y, 3))``, where `3` is for `(r, g, b)` channels

    * ``ti.Vector(3, shape=(x, y))`` (see :ref:`vector`)

    * ``np.ndarray(shape=(x, y))``

    * ``np.ndarray(shape=(x, y, 3))``


    The data type of ``img`` must be one of:

    * float32, clamped into [0, 1]

    * float64, clamped into [0, 1]

    * uint8, range [0, 255]

    * uint16, range [0, 65535]

    * uint32, range [0, UINT_MAX]


.. function:: gui.circle(pos, color = 0xFFFFFF, radius = 1)

    :parameter gui: (GUI) the window object
    :parameter pos: (tuple of 2) the position of circle
    :parameter color: (optional, RGB hex) color to fill the circle
    :parameter radius: (optional, scalar) the radius of circle

    Draw a solid circle.


.. function:: gui.circles(pos, color = 0xFFFFFF, radius = 1)

    :parameter gui: (GUI) the window object
    :parameter pos: (np.array) the position of circles
    :parameter color: (optional, RGB hex or np.array of uint32) color(s) to fill circles
    :parameter radius: (optional, scalar) the radius of circle

    Draw solid circles.

.. note::

    If ``color`` is a numpy array, circle at ``pos[i]`` will be colored with ``color[i]``, therefore it must have the same size with ``pos``.


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


.. function:: gui.get_event(a, ...)

    :parameter gui: (GUI)
    :parameter a: (optional, EventFilter) filter out matched events
    :return: (bool) ``False`` if there is no pending event, vise versa

    Try to pop a event from the queue, and store it in ``gui.event``.

    For example:

    ::

        while gui.get_event():
            print('Event key', gui.event.key)


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
