---
sidebar_position: 1

---

# GUI system

Taichi has a built-in GUI system to help users visualize results.

## Create a window

`ti.GUI(name, res)` creates a window. If `res` is scalar, then width will be equal to height.

The following codes show how to create a window of resolution `640x360`:

```python
gui = ti.GUI('Window Title', (640, 360))
```

:::note

If you are running Taichi on a machine without a GUI environment, consider setting `show_gui` to `False`:

```python
gui = ti.GUI('Window Title', (640, 360), show_gui=False)

while gui.running:
    ...
    gui.show(f'{gui.frame:06d}.png')  # save a series of screenshot
```

:::

## Display a window

`gui.show(filename)` helps display a window. If `filename` is specified, a screenshot will be saved to the file specified by the name. For example, the following saves frames of the window to `.png`s:

    for frame in range(10000):
        render(img)
        gui.set_image(img)
        gui.show(f'{frame:06d}.png')



## Paint on a window

`gui.set_image(pixels)` sets an image to display on the window.

The image pixels are set from the values of `img[i, j]`, where `i` indicates the horizontal coordinates (from left to right) and `j` the vertical coordinates (from bottom to top).

If the window size is `(x, y)`, then `img` must be one of:

- `ti.field(shape=(x, y))`, a gray-scale image

- `ti.field(shape=(x, y, 3))`, where `3` is for `(r, g, b)` channels

- `ti.field(shape=(x, y, 2))`, where `2` is for `(r, g)` channels

- `ti.Vector.field(3, shape=(x, y))` `(r, g, b)` channels on each
  component

- `ti.Vector.field(2, shape=(x, y))` `(r, g)` channels on each component

- `np.ndarray(shape=(x, y))`

- `np.ndarray(shape=(x, y, 3))`

- `np.ndarray(shape=(x, y, 2))`

The data type of `img` must be one of:

- `uint8`, range `[0, 255]`

- `uint16`, range `[0, 65535]`

- `uint32`, range `[0, 4294967295]`

- `float32`, range `[0, 1]`

- `float64`, range `[0, 1]`



## Convert RGB to Hex

`ti.rgb_to_hex(rgb)` can convert a (R, G, B) tuple of floats into a single integer value, e.g.,

```python
rgb = (0.4, 0.8, 1.0)
hex = ti.rgb_to_hex(rgb)  # 0x66ccff

rgb = np.array([[0.4, 0.8, 1.0], [0.0, 0.5, 1.0]])
hex = ti.rgb_to_hex(rgb)  # np.array([0x66ccff, 0x007fff])
```

The return values can be used in GUI drawing APIs.



## Event processing

Every event have a key and type.

_Event type_ is the type of event, for now, there are just three type of event:

    ti.GUI.RELEASE  # key up or mouse button up
    ti.GUI.PRESS    # key down or mouse button down
    ti.GUI.MOTION   # mouse motion or mouse wheel

_Event key_ is the key that you pressed on keyboard or mouse, can be one of:

    # for ti.GUI.PRESS and ti.GUI.RELEASE event:
    ti.GUI.ESCAPE  # Esc
    ti.GUI.SHIFT   # Shift
    ti.GUI.LEFT    # Left Arrow
    'a'            # we use lowercase for alphabet
    'b'
    ...
    ti.GUI.LMB     # Left Mouse Button
    ti.GUI.RMB     # Right Mouse Button

    # for ti.GUI.MOTION event:
    ti.GUI.MOVE    # Mouse Moved
    ti.GUI.WHEEL   # Mouse Wheel Scrolling

A _event filter_ is a list combined of _key_, _type_ and _(type, key)_ tuple, e.g.:

```python
# if ESC pressed or released:
gui.get_event(ti.GUI.ESCAPE)

# if any key is pressed:
gui.get_event(ti.GUI.PRESS)

# if ESC pressed or SPACE released:
gui.get_event((ti.GUI.PRESS, ti.GUI.ESCAPE), (ti.GUI.RELEASE, ti.GUI.SPACE))
```

`gui.running` can help check the state of the window. `ti.GUI.EXIT` occurs when you click on the close (X) button of a window.
 `gui.running` will obtain `False` when the GUI is being closed.

For example, loop until the close button is clicked:

    while gui.running:
        render()
        gui.set_image(pixels)
        gui.show()

You can also close the window by manually setting `gui.running` to`False`:

    while gui.running:
        if gui.get_event(ti.GUI.ESCAPE):
            gui.running = False

        render()
        gui.set_image(pixels)
        gui.show()

`gui.get_event(a, ...)` tries to pop an event from the queue, and stores it into `gui.event`.

For example:

    if gui.get_event():
        print('Got event, key =', gui.event.key)

For example, loop until ESC is pressed:

    gui = ti.GUI('Title', (640, 480))
    while not gui.get_event(ti.GUI.ESCAPE):
        gui.set_image(img)
        gui.show()

`gui.get_events(a, ...)` is basically the same as `gui.get_event`, except that it returns a generator of events instead of storing into `gui.event`:

    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            do_something()
        elif e.key in ['a', ti.GUI.LEFT]:
            ...

`gui.is_pressed(key, ...)` can detect the keys you pressed. It must be used together with `gui.get_event`, or it won't be updated! For
example:

    while True:
        gui.get_event()  # must be called before is_pressed
        if gui.is_pressed('a', ti.GUI.LEFT):
            print('Go left!')
        elif gui.is_pressed('d', ti.GUI.RIGHT):
            print('Go right!')

`gui.get_cursor_pos()` can return current cursor position within the window. For example:

    mouse_x, mouse_y = gui.get_cursor_pos()

`gui.fps_limit` sets the FPS limit for a window. For example, to cap FPS at 24, simply use `gui.fps_limit = 24`. This helps reduce the overload on your hardware especially when you're using OpenGL on your integrated GPU which could make desktop slow to response.



## GUI Widgets

Sometimes it's more intuitive to use widgets like slider or button to control the program variables instead of using chaotic keyboard bindings. Taichi GUI provides a set of widgets for that reason:

For example:

    radius = gui.slider('Radius', 1, 50)

    while gui.running:
        print('The radius now is', radius.value)
        ...
        radius.value += 0.01
        ...
        gui.show()



## Image I/O

`ti.imwrite(img, filename)` can export a `np.ndarray` or Taichi field (`ti.Matrix.field`,  `ti.Vector.field`, or `ti.field`) to a specified location `filename`.

Same as `ti.GUI.show(filename)`, the format of the exported image is determined by **the suffix of** `filename` as well. Now `ti.imwrite` supports exporting images to `png`, `img` and `jpg` and we recommend using `png`.

Please make sure that the input image has **a valid shape**. If you want to export a grayscale image, the input shape of field should be `(height, weight)` or `(height, weight, 1)`. For example:

```python
import taichi as ti

ti.init()

shape = (512, 512)
type = ti.u8
pixels = ti.field(dtype=type, shape=shape)

@ti.kernel
def draw():
    for i, j in pixels:
        pixels[i, j] = ti.random() * 255    # integars between [0, 255] for ti.u8

draw()

ti.imwrite(pixels, f"export_u8.png")
```

Besides, for RGB or RGBA images, `ti.imwrite` needs to receive a field which has shape `(height, width, 3)` and `(height, width, 4)` individually.

Generally the value of the pixels on each channel of a `png` image is an integer in \[0, 255\]. For this reason, `ti.imwrite` will **cast fields** which has different data types all **into integers between \[0, 255\]**. As a result, `ti.imwrite` has the following requirements for different data types of input fields:

- For float-type (`ti.f16`, `ti.f32`, etc) input fields, **the value of each pixel should be float between \[0.0, 1.0\]**. Otherwise `ti.imwrite` will first clip them into \[0.0, 1.0\]. Then they are multiplied by 256 and casted to integers ranging from \[0, 255\].
- For int-type (`ti.u8`, `ti.u16`, etc) input fields, **the value of each pixel can be any valid integer in its own bounds**. These integers in this field will be scaled to \[0, 255\] by being divided over the upper bound of its basic type accordingly.

Here is another example:

```python
import taichi as ti

ti.init()

shape = (512, 512)
channels = 3
type = ti.f32
pixels = ti.Matrix.field(channels, dtype=type, shape=shape)

@ti.kernel
def draw():
    for i, j in pixels:
        for k in ti.static(range(channels)):
            pixels[i, j][k] = ti.random()   # floats between [0, 1] for ti.f32

draw()

ti.imwrite(pixels, f"export_f32.png")
```
