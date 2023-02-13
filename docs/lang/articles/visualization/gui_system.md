---
sidebar_position: 1
---

# GUI System

Taichi has a built-in GUI system for visualizing simulation data in data containers like Taichi fields or NumPy ndarrays. It also has limited support for drawing primitive geometries.

## Create and display a window

The following code creates a `640x360` window with a "Hello World!" title, and displays it by calling `gui.show()`:

```python
gui = ti.GUI('Hello World!', (640, 360))
while gui.running:
    gui.show()
```

In order to ensure a continuous visual display, it is imperative to put the call to `gui.show()` within a `while` loop. Otherwise, the window will only briefly appear before being automatically closed.


## Close the window

You can set `gui.running=False` in the `while` loop to close the GUI:

```python
gui = ti.GUI('Window Title', (640, 360))
while gui.running:
    if some_events_happend:
        gui.running = False
    gui.show()
```

## Coordinate system

Each window in the system is constructed based on a coordinate system, with the origin positioned at the lower-left corner. The positive x-axis extends towards the right, while the positive y-axis extends upwards.

## Display a field or ndarray

To display a Taichi field or a NumPy ndarray, call `gui.set_image()`. The method accepts both types as input.

```python
image = ti.Vector.field(3, ti.f32, shape=(640, 480))
while gui.running:
    gui.set_image(image)
    gui.show()
```

Because Taichi field is a *global* data container, if the vector field `image` is updated between the `while` loops, the GUI window refreshes to display the latest image.

:::caution IMPORTANT

Ensure that the shape of the input matches the resolution of the GUI window.

:::

### Zero-copying frame buffer

In each iteration of the invocation of the `gui.set_image()` method, the GUI framework performs a conversion of the image data into a format suitable for display and transfers the converted data to the window's frame buffer. This results in a substantial increase in processing overhead when the window size is substantial, which hinders the ability to attain high frame rates (FPS).

In instances where the sole purpose of calling the `set_image()` method is to display an image, and no additional drawing operations are required, it is possible to enable the `fast_gui` mode for improved performance. This mode enables Taichi GUI to directly write the image data to the frame buffer, thereby eliminating the need for additional data copying and significantly enhancing the frame rate (FPS).

```python
gui = ti.GUI(res, title, fast_gui=True)
```

In order to ensure that the `fast_gui` mode operates effectively, it is crucial to verify that the data passed to the `gui.set_image()` method is in a format that is suitable for display. Specifically, if the data is represented as a Taichi field, it should be of one of the following formats:

- a vector field `ti.field(3, dtype, shape)` compatible with RGB format.
- a vector field `ti.field(4, dtype, shape)`  compatible with RGBA format.

It is important to note that the `dtype` attribute of the Taichi field must be set to `ti.f32`, `ti.f64`, or `ti.u8`.


## Draw on a window

In addition to image display, Taichi's GUI system also supports the rendering of simple geometric shapes such as lines, triangles, rectangles, circles, and texts.

The `pos` parameter in each of the drawing methods accommodates Taichi fields or NumPy arrays, but not Python lists of primitive data types. The elements within the arrays are represented as pairs of floating-point numbers, with values ranging from 0.0 to 1.0, denoting the relative positions of the geometries. For instance:

- `(0.0, 0.0)`: the lower-left corner of the window.
- `(1.0, 1.0)`: the upper-right corner of the window.

The following code snippet demonstrates the rendering of 50 circles with a radius of 5, each with a color randomly assigned based on the values in indices, an integer array with the same size as `pos`:

```python
import numpy as np
pos = np.random.random((50, 2))
# Create an array of 50 integer elements whose values are randomly 0, 1, 2
# 0 corresponds to 0x068587
# 1 corresponds to 0xED553B
# 2 corresponds to 0xEEEEF0
indices = np.random.randint(0, 2, size=(50,))
gui = ti.GUI("circles", res=(400, 400))
while gui.running:
    gui.circles(pos, radius=5, palette=[0x068587, 0xED553B, 0xEEEEF0], palette_indices=indices)
    gui.show()
```

![gui-circles](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/doc/gui-circles.png)

The following code snippet demonstrates the rendering of five blue line segments, each with a width of 2, with `X` and `Y` denoting the starting and ending points respectively:

```python
import numpy as np
X = np.random.random((5, 2))
Y = np.random.random((5, 2))
gui = ti.GUI("lines", res=(400, 400))
while gui.running:
    gui.lines(begin=X, end=Y, radius=2, color=0x068587)
    gui.show()
```

![gui-lines](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/doc/gui-lines.png)

The following code snippet demonstrates the rendering of two orange triangles, with `X`, `Y`, and `Z` denoting the three vertices of each triangle:

```python
import numpy as np
X = np.random.random((2, 2))
Y = np.random.random((2, 2))
Z = np.random.random((2, 2))
gui = ti.GUI("triangles", res=(400, 400))
while gui.running:
    gui.triangles(a=X, b=Y, c=Z, color=0xED553B)
    gui.show()
```

![gui-triangles](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/doc/gui-triangles.png)


## Event handling

In addition to its rendering capabilities, Taichi's GUI system also offers a range of methods for mouse and keyboard input control. Input events are categorized into three categories:

```python
ti.GUI.RELEASE  # key up or mouse button up
ti.GUI.PRESS    # key down or mouse button down
ti.GUI.MOTION   # mouse motion or mouse wheel
```

*Event key*: refers to the key that you press from your keyboard or mouse. It can be one of:

```python
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
```

*Event filter*: refers to a combined list of *key*, *type*, and (*type*, *key*) tuple. For example:

```python
# if ESC pressed or released:
gui.get_event(ti.GUI.ESCAPE)

# if any key is pressed:
gui.get_event(ti.GUI.PRESS)

# if ESC is pressed or SPACE is released:
gui.get_event((ti.GUI.PRESS, ti.GUI.ESCAPE), (ti.GUI.RELEASE, ti.GUI.SPACE))
```

`gui.get_event()`: retrieves and removes the next event from the queue and saves it to `gui.event`. For example:

```python
if gui.get_event():
    print('Got event, key =', gui.event.key)
```

The following code defines that the `while` loop goes on until **ESC** is pressed:

```python
gui = ti.GUI('Title', (640, 480))
while not gui.get_event(ti.GUI.ESCAPE):
    gui.set_image(img)
    gui.show()
```

`ggui.is_pressed(): identifies currently pressed keys. As demonstrated in the following code snippet, it must be used in conjunction with `gui.get_event()`, as it will not be updated otherwise. For example:

```python
while True:
    gui.get_event()  # must be called before is_pressed
    if gui.is_pressed('a', ti.GUI.LEFT):
        print('Go left!')
    elif gui.is_pressed('d', ti.GUI.RIGHT):
        print('Go right!')
```

This method returns `True` if the specified `event_filter` is matched, otherwise returns `False`.

:::caution

It is crucial to call `gui.get_event()` before calling `gui.is_pressed()`, as the latter will not have the desired effect if not used in the proper order.
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

#### Retrieve cursor position

The `gui.get_cursor_pos()` method returns the current position of the cursor in the GUI window as a pair of normalized `x` and `y` float values in the range `[0.0, 1.0]`. This information can be used to build interactive applications that respond to cursor movement and user input.

```python
mouse_x, mouse_y = gui.get_cursor_pos()
```

## GUI Widgets

The Taichi GUI system also features a suite of widgets, such as `slider()`, `label()`, and `button()`, that enable you to create a customized control interface. The following code provides a simple example:

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
