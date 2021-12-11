---
sidebar_position: 1

---

# GUI system

Taichi has a built-in GUI system to help users visualize results.

## Create a window

[`ti.GUI(name, res)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=gui%20gui#taichi.misc.gui.GUI)
creates a window.

The following code show how to create a window of resolution `640x360`:

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

[`gui.show(filename)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=show#taichi.misc.gui.GUI.show)
helps display a window. If `filename` is specified, a screenshot will be saved to the path. For example, the following saves frames of the window to `.png`s:

    for frame in range(10000):
        render(img)
        gui.set_image(img)
        gui.show(f'{frame:06d}.png')



## Paint on a window
Taichi's GUI supports painting simple geometric objects, such as lines, triangles, rectangles, circles, and text.

:::note

The position parameter of every drawing API expects input of 2-element tuples,
whose values are the relative position of the object range from 0.0 to 1.0.
(0.0, 0.0) stands for the lower left corner of the window, and (1.0, 1.0) stands for the upper right corner.

Acceptable input for positions are taichi fields or numpy arrays. Primitive arrays in python are NOT acceptable.

For simplicity, we use numpy arrays in the examples below.

:::

:::tip

For detailed API description, please click on the API code. For instance, click on
`gui.get_image()` to see the description to get a GUI images.

:::

[`gui.set_image(pixels)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=set_image#taichi.misc.gui.GUI.set_image)
sets an image to display on the window.

The image pixels are set from the values of `img[i, j]`, where `i` indicates the horizontal coordinates (from left to right) and `j` the vertical coordinates (from bottom to top).

If the window size is `(x, y)`, then `img` must be one of:

- `ti.field(shape=(x, y))`, a gray-scale image

- `ti.field(shape=(x, y, 3))`, where `3` is for `(r, g, b)` channels

- `ti.field(shape=(x, y, 2))`, where `2` is for `(r, g)` channels

- `ti.Vector.field(3, shape=(x, y))` `(r, g, b)` channels on each component

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

[`gui.get_image()`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=get_image#taichi.misc.gui.GUI.get_image)
gets the 4-channel (RGBA) image shown in the current GUI system.

[`gui.circle(pos)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=circle#taichi.misc.gui.GUI.circle)
draws one solid circle.

The color and radius of circles can be further specified with additional parameters.

[`gui.circles(pos)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=circles#taichi.misc.gui.GUI.circles)
draws solid circles.

The color and radius of circles can be further specified with additional parameters. For a single color, use the `color` parameter.
For multiple colors, use `palette` and `palette_indices` instead.

:::note

The unit of raduis in GUI APIs is number of pixels.

:::

For examples:
```python
gui.circles(pos, radius=3, color=0x068587)
```
draws circles all with radius of 1.5 and blue color positioned at pos array.

![circles](../static/assets/circles.png)
```python
gui.circles(pos, radius=3, palette=[0x068587, 0xED553B, 0xEEEEF0], palette_indices=material)
```
draws circles with radius of 1.5 and three different colors differed by `material`, an integer array with the same size as
`pos`. Each integer in `material` indicates which color the associated circle use (i.e. array [0, 1, 2] indicates these three
circles are colored separately by the first, second, and third color in `palette`.

![circles](../static/assets/colored_circles.png)

[`gui.line(begin, end)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=line#taichi.misc.gui.GUI.line)
draws one line.

The color and radius of lines can be further specified with additional parameters.

[`gui.lines(begin, end)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=line#taichi.misc.gui.GUI.lines)
draws lines.

`begin` and `end` both require input of positions.

The color and radius of lines can be further specified with additional parameters.

For example:
```python
gui.lines(begin=X, end=Y, radius=2, color=0x068587)
```
draws line segments from X positions to Y positions with width of 2 and color in light blue.

![lines](../static/assets/lines.png)

[`gui.triangle(a, b, c)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=triangle#taichi.misc.gui.GUI.triangle)
draws one solid triangle.

The color of triangles can be further specified with additional parameters.

[`gui.triangles(a, b, c)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=triangles#taichi.misc.gui.GUI.triangles)
draws solid triangles.

The color of triangles can be further specified with additional parameters.

For example:
```python
gui.triangles(a=X, b=Y, c=Z, color=0xED553B)
```
draws triangles with color in red and three points positioned at X, Y, and Z.

![triangles](../static/assets/triangles.png)

[`gui.rect(topleft, bottomright)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=rect#taichi.misc.gui.GUI.rect)
draws a hollow rectangle.

The color and radius of the stroke of rectangle can be further specified with additional parameters.

For example:
```python
gui.rect([0, 0], [0.5, 0.5], radius=1, color=0xED553B)
```
draws a rectangle of top left corner at [0, 0] and bottom right corner at [0.5, 0.5], with stroke of radius of 1 and color in red.

![rect](../static/assets/rect.png)

[`gui.arrows(origin, direction)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=arrows#taichi.misc.gui.GUI.arrows)
draws arrows.

`origin` and `direction` both require input of positions. `origin` refers to the positions of arrows' origins, `direction`
refers to the directions where the arrows point to relative to their origins.

The color and radius of arrows can be further specified with additional parameters.

For example:
```python
x = numpy.array([[0.1, 0.1], [0.9, 0.1]])
y = numpy.array([[0.3, 0.3], [-0.3, 0.3]])
gui.arrows(x, y, radius=1, color=0xFFFFFF)
```
draws two arrow originated at [0.1, 0.1], [0.9, 0.1] and pointing to [0.3, 0.3], [-0.3, 0.3] with radius of 1 and color in white.

![arrows](../static/assets/arrows.png)

[`gui.arrow_field(direction)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=arrow_field#taichi.misc.gui.GUI.arrow_field)
draws a field of arrows.

The `direction` requires a field of `shape=(col, row, 2)` where `col` refers to the number of columns of arrow field and `row`
refers to the number of rows of arrow field.

The color and bound of arrow field can be further specified with additional parameters.

For example:
```python
gui.arrow_field(x, bound=0.5, color=0xFFFFFF) # x is a field of shape=(5, 5, 2)
```
draws a 5 by 5 arrows pointing to random directions.

![arrow_field](../static/assets/arrow_field.png)

[`gui.point_field(radius)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=point_field#taichi.misc.gui.GUI.point_field)
draws a field of points.

The `radius` requires a field of `shape=(col, row)` where `col` refers to the number of columns of arrow field and `row`
refers to the number of rows of arrow field.

The color and bound of point field can be further specified with additional parameters.

For example:
```python
x = numpy.array([[3, 5, 7, 9], [9, 7, 5, 3], [6, 6, 6, 6]])
gui.point_field(radius=x, bound=0.5, color=0xED553B)
```
draws a 3 by 4 point field of radius stored in the array.

![point_field](../static/assets/point_field.png)

[`gui.text(content, pos)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=text#taichi.misc.gui.GUI.text)
draws a line of text on screen.

The font size and color of text can be further specified with additional parameters.

## RGB & Hex conversion.

[`ti.hex_to_rgb(hex)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=hex_to_rgb#taichi.misc.gui.hex_to_rgb)
can convert a single integer value to a (R, G, B) tuple of floats.

[`ti.rgb_to_hex(rgb)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=rgb#taichi.misc.gui.rgb_to_hex)
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

[`gui.running`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=running#taichi.misc.gui.GUI.running)
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

[`gui.get_event(a, ...)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=get_event#taichi.misc.gui.GUI.get_event)
tries to pop an event from the queue, and stores it into `gui.event`.

For example:

    if gui.get_event():
        print('Got event, key =', gui.event.key)

For example, loop until ESC is pressed:

    gui = ti.GUI('Title', (640, 480))
    while not gui.get_event(ti.GUI.ESCAPE):
        gui.set_image(img)
        gui.show()

[`gui.get_events(a, ...)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=get_event#taichi.misc.gui.GUI.get_events)
is basically the same as `gui.get_event`, except that it returns a generator of events instead of storing into `gui.event`:

    for e in gui.get_events():
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            do_something()
        elif e.key in ['a', ti.GUI.LEFT]:
            ...

[`gui.is_pressed(key, ...)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=is_pressed#taichi.misc.gui.GUI.is_pressed)
can detect the keys you pressed. It must be used together with `gui.get_event`, or it won't be updated! For
example:

    while True:
        gui.get_event()  # must be called before is_pressed
        if gui.is_pressed('a', ti.GUI.LEFT):
            print('Go left!')
        elif gui.is_pressed('d', ti.GUI.RIGHT):
            print('Go right!')

:::caution

`gui.is_pressed()` must be used together with `gui.get_event()`, or it won't be updated!

:::

For example:

```python
while True:
    gui.get_event() # must be called before is_pressed
    if gui.is_pressed('a', ti.GUI.LEFT):
        print('Go left!')
    elif gui.is_pressed('d', ti.GUI.RIGHT):
        print('Go right!')
```

[`gui.get_cursor_pos()`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=get_cursor#taichi.misc.gui.GUI.get_cursor_pos)
can return current cursor position within the window. For example:

    mouse_x, mouse_y = gui.get_cursor_pos()

[`gui.fps_limit`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=fps#taichi.misc.gui.GUI.fps_limit)
sets the FPS limit for a window. For example, to cap FPS at 24, simply use `gui.fps_limit = 24`. This helps reduce the overload on your hardware especially when you're using OpenGL on your integrated GPU which could make desktop slow to response.



## GUI Widgets

Sometimes it's more intuitive to use widgets like slider or button to control the program variables instead of using chaotic keyboard bindings. Taichi GUI provides a set of widgets for that reason:

[`gui.slider(text, min, max)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=slider#taichi.misc.gui.GUI.slider)
creates a slider following the text `{text}: {value:.3f}`.

[`gui.label(text)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=label#taichi.misc.gui.GUI.label)
displays the label as: `{text}: {value:.3f}`.

[`gui.button(text)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=button#taichi.misc.gui.GUI.button)
creates a button with text on it.

For example:
```python
import taichi as ti

gui = ti.GUI('GUI widgets')

radius = gui.slider('Radius', 1, 50, step=1)
xcoor = gui.label('X-coordinate')
okay = gui.button('OK')

xcoor.value = 0.5
radius.value = 10

while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'a':
            xcoor.value -= 0.05
        elif e.key == 'd':
            xcoor.value += 0.05
        elif e.key == 's':
            radius.value -= 1
        elif e.key == 'w':
            radius.value += 1
        elif e.key == okay:
            print('OK clicked')

    gui.circle((xcoor.value, 0.5), radius=radius.value)
    gui.show()
```



## Image I/O

[`ti.imwrite(img, filename)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=imwrite#taichi.misc.image.imwrite)
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
        pixels[i, j] = ti.random() * 255    # integers between [0, 255] for ti.u8

draw()

ti.imwrite(pixels, f"export_u8.png")
```

Besides, for RGB or RGBA images, `ti.imwrite` needs to receive a field which has shape `(height, width, 3)` and `(height, width, 4)` individually.

Generally the value of the pixels on each channel of a `png` image is an integer in \[0, 255\]. For this reason, `ti.imwrite` will **cast fields** which has different data types all **into integers between \[0, 255\]**. As a result, `ti.imwrite` has the following requirements for different data types of input fields:

- For float-type (`ti.f16`, `ti.f32`, etc.) input fields, **the value of each pixel should be float between \[0.0, 1.0\]**. Otherwise `ti.imwrite` will first clip them into \[0.0, 1.0\]. Then they are multiplied by 256 and cast to integers ranging from \[0, 255\].
- For int-type (`ti.u8`, `ti.u16`, etc.) input fields, **the value of each pixel can be any valid integer in its own bounds**. These integers in this field will be scaled to \[0, 255\] by being divided over the upper bound of its basic type accordingly.

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

[`ti.imread(filename)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=imread#taichi.misc.image.imread)
loads an image from the target filename and returns it as a `np.ndarray(dtype=np.uint8)`.
Each value in this returned field is an integer in [0, 255].

[`ti.imshow(img, windname)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=imshow#taichi.misc.image.imshow)
creates an instance of ti.GUI and show the input image on the screen. It has the same logic as `ti.imwrite` for different data types.

[`ti.imresize(img, w)`](https://api-docs.taichi.graphics/src/taichi.misc.html?highlight=imresize#taichi.misc.image.imresize)
resizes the img specified.

## Zero-copying frame buffer
When the GUI resolution (window size) is large, it sometimes becomes difficult to achieve 60 FPS even without any kernel
invocations between two frames.

This is mainly due to the copy overhead, where Taichi GUI needs to copy the image buffer from one place to another.
This process is necessary for the 2D drawing functions, such as `gui.circles`, to work. The larger the image shape is,
the larger the overhead.

Fortunately, sometimes your program only needs `gui.set_image` alone. In such cases, you can enable the `fast_gui` option
for better performance. This mode allows Taichi GUI to directly write the image data to the frame buffer without additional
copying, resulting in a much better FPS.

```python
gui = ti.GUI(res, title, fast_gui=True)
```

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
