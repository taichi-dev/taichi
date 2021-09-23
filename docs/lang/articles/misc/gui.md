---
sidebar_position: 1

---

# GUI system

Taichi has a built-in GUI system to help users visualize results.

## Create a window

[`ti.GUI(name, res)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=gui%20gui#taichi.misc.gui.GUI)
creates a window. If `res` is scalar, then width will be equal to height.

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

[`gui.show(filename)`](TODO: Link here)
helps display a window. If `filename` is specified, a screenshot will be saved to the file specified by the name. For example, the following saves frames of the window to `.png`s:

    for frame in range(10000):
        render(img)
        gui.set_image(img)
        gui.show(f'{frame:06d}.png')



## Paint on a window
Taichi's GUI supports painting simple geometrix objects, such as lines, triangles, rectangles, circles, and text.

:::note

The position parameter `pos` expects an input of a 2-element tuple, whose values are the relative position of the object. (0.0, 0.0) stands for the lower left corner of the window, and (1.0, 1.0) stands for the upper right corner.

:::

[`gui.set_image(pixels)`](TODO: Link here)
sets an image to display on the window.

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

:::note

When using `float32` or `float64` as the data type, `img` entries will be clipped into range [0, 1] for display.

:::

[`gui.get_image()`](TODO)
gets the 4-channel (RGBA) image shown in the current GUI system.

[`gui.circles(pos)`](TODO)
draws solid circles.

[`gui.lines(begin, end)`](TODO)
draws lines.

[`gui.triangles(a, b, c)`](TODO)
draws solid triangles.

[`gui.rect(topleft, bottomright)`](TODO)
draws a rectangle.

[`gui.text(content, pos)`](TODO)
draws a line of text on screen.

## Convert RGB to Hex

[`ti.rgb_to_hex(rgb)`](TODO)
can convert a (R, G, B) tuple of floats into a single integer value, e.g.,

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

[`gui.running`](TODO)
can help check the state of the window. `ti.GUI.EXIT` occurs when you click on the close (X) button of a window.
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

[`gui.get_event(a, ...)`](TODO)
tries to pop an event from the queue, and stores it into `gui.event`.

For example:

    if gui.get_event():
        print('Got event, key =', gui.event.key)

For example, loop until ESC is pressed:

    gui = ti.GUI('Title', (640, 480))
    while not gui.get_event(ti.GUI.ESCAPE):
        gui.set_image(img)
        gui.show()

[`gui.get_events(a, ...)`](TODO)
is basically the same as `gui.get_event`, except that it returns a generator of events instead of storing into `gui.event`:

    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            do_something()
        elif e.key in ['a', ti.GUI.LEFT]:
            ...

[`gui.is_pressed(key, ...)`](TODO)
can detect the keys you pressed. It must be used together with `gui.get_event`, or it won't be updated! For
example:

    while True:
        gui.get_event()  # must be called before is_pressed
        if gui.is_pressed('a', ti.GUI.LEFT):
            print('Go left!')
        elif gui.is_pressed('d', ti.GUI.RIGHT):
            print('Go right!')

:::caution

Must be used together with `gui.get_event`, or it won't be updated!

For example:

```python
while True:
    gui.get_event() # must be called before is_pressed
    if gui.is_pressed('a', ti.GUI.LEFT):
        print('Go left!')
    elif gui.is_pressed('d', ti.GUI.RIGHT):
        print('Go right!')
```

:::

[`gui.get_cursor_pos()`](TODO)
can return current cursor position within the window. For example:

    mouse_x, mouse_y = gui.get_cursor_pos()

[`gui.fps_limit`](TODO)
sets the FPS limit for a window. For example, to cap FPS at 24, simply use `gui.fps_limit = 24`. This helps reduce the overload on your hardware especially when you're using OpenGL on your integrated GPU which could make desktop slow to response.



## GUI Widgets

Sometimes it's more intuitive to use widgets like slider or button to control the program variables instead of using chaotic keyboard bindings. Taichi GUI provides a set of widgets for that reason:

[`gui.slider(text, min, max)`](TODO)
creates a slider following the text `{text}: {value:.3f}`.

[`gui.label(text)`](TODO)
displays the label as: `{text}: {value:.3f}`.

[`gui.button(text)`](TODO)
creates a button with text on it.

For example:
```python
radius = gui.slider('Radius', 1, 50)

while gui.running:
    print('The radius now is', radius.value)
    ...
    radius.value += 0.01
    ...
    gui.show()
```



## Image I/O

[`ti.imwrite(img, filename)`](TODO)
can export a `np.ndarray` or Taichi field (`ti.Matrix.field`,  `ti.Vector.field`, or `ti.field`) to a specified location `filename`.

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

[`ti.imread(filename)`](TODO)
loads an image from the target filename and returns it as a `np.ndarray(dtype=np.uint8)`.
Each value in this returned field is an integer in [0, 255].

[`ti.imshow(img, windname)`](TODO)
creates an instance of ti.GUI and show the input image on the screen. It has the same logic as `ti.imwrite` for different datatypes.

[`ti.imresize(img, w)`](TODO)
resizes the img specified.

### Zero-copying frame buffer
When the GUI resolution (window size) is large, it sometimes becomes difficult to achieve 60 FPS even without any kernel
invocations between two frames.

This is mainly due to the copy overhead, where Taichi GUI needs to copy the image buffer from one place to another.
This copying is necessary for the 2D drawing functions, such as `gui.circles`, to work. The larger the image shape is,
the larger the overhead.

Fortunately, sometimes your program only needs `gui.set_image` alone. In such cases, you can enable the `fast_gui` option
for better performance. This mode allows Taichi GUI to directly write the image data to the frame buffer without additional
copying, resulting in a much better FPS.

`gui = ti.GUI(res, title, fast_gui=True)`

:::note

Because of the zero-copying mechanism, the image passed into `gui.set_image` must already be in the display-compatible
format. That is, this field must either be a `ti.Vector(3)` (RGB) or a `ti.Vector(4)` (RGBA). In addition, each channel
must be of type `ti.f32`, `ti.f64` or `ti.u8`.

:::

:::note

If possible, consider enabling this option, especially when `fullscreen=True`.

:::

:::caution

Despite the performance boost, it has many limitations as trade off:

`gui.set_image` is the only available paint API in this mode.

`gui.set_image` will only take Taichi 3D or 4D vector fields (RGB or RGBA) as input.

:::