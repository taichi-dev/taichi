---
sidebar_position: 1

---

# A New UI system: GGUI

:::caution
GGUI is currently supported on the x64 and CUDA backends for the system Windows and Linux.

You also need to install the Vulkan environment: [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home).

:::

A new UI system has been added to Taichi in version `v0.8.0`. The new GUI system uses GPU for rendering, enabling it to be much faster and to render 3d scenes. For these reasons, this new system is sometimes referred to as GGUI. This doc describes the APIs provided.

Apart from this doc, a good way of getting familiarized with GGUI is to look at the examples. Please checkout the examples provided in  [`examples/ggui_examples`](https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples/ggui_examples).

## Creating a window

`ti.ui.Window(name, res)` creates a window.

```python
window = ti.ui.Window('Window Title', (640, 360))
```

There are three types of objects that can be displayed on a `ti.ui.Window`:

* 2D Canvas, which can be used to draw simple 2D geometries such as circles, triangles, etc.
* 3D Scene, which can be used to render 3D meshes and particles, with a configurable camera and light sources.
* Immediate mode GUI components, e.g., buttons, textboxes, etc.

## 2D Canvas

### Creating a canvas

```python
canvas = window.get_canvas()
```
this retrieves a `Canvas` object that covers the entire window.

### Drawing on the canvas

```python
canvas.set_background_color(color)
canvas.triangles(vertices, color, indices, per_vertex_color)
canvas.circles(vertices, radius, color, per_vertex_color)
canvas.lines(vertices, width, indices, color, per_vertex_color)
canvas.set_image(image)
```

The arguments `vertices`, `indices`, `per_vertex_color`, and `image` are all expected to be Taichi fields. If `per_vertex_color` is provided, `color` will be ignored.

The positions/centers of geometries will be represented as floats between 0 and 1, which indicate relative positions on the canvas. For `circles` and `lines`, the `radius` and `width` arguments are relative to the height of the window.

The canvas will be cleared after every frame. You should call these methods within the render loop.


## 3D Scene

### Creating a scene
```python
scene = ti.ui.Scene()
```
### Configuring camera
```python
camera = ti.ui.make_camera()
camera.position(pos)
camera.lookat(pos)
camera.up(dir)
camera.projection_mode(mode)
scene.set_camera(camera)
```


### Configuring light sources
#### Adding a point light
```python
scene.point_light(pos, color)
```
Note that `point_light` method needs to be called every frame. Similar to the `canvas` methods, you should call this within your render loop.


### 3d Geometries
```python
scene.mesh(vertices, indices, normals, color, per_vertex_color)
scene.particles(vertices, radius, color, per_vertex_color)
```

The arguments `vertices`, `indices`, `per_vertex_color`, and `image` are all expected to be Taichi fields. If `per_vertex_color` is provided, `color` will be ignored.

The positions/centers of geometries should be in the world-space coordinates.

:::note

If a mesh has `num` triangles, the `indices` should be a 1D scalar field with a shape of `num * 3` instead of a vector field.

The `normals` parameter for `scene.mesh` is optional.
:::


### Rendering the scene
A scene can be rendered on a canvas.
```python
canvas.scene(scene)
```

## GUI components

The support for GUI components will closely follow the Dear IMGUI APIs:

```python
window.GUI.begin(name, x, y, width, height)
window.GUI.text(text)
is_clicked = window.GUI.button(name)
new_value = window.GUI.slider_float(name, old_value, min_value, max_value)
new_color = window.GUI.color_edit_3(name, old_color)
window.GUI.end()
```

## Showing a window

```python
...
window.show()
```
Call this method at the very end of the render loop for each frame.

## User Input Processing
To obtain the events that have occurred since the previous poll:

```python
events = window.get_events()
```

Each `event` in `events` is an instance of `ti.ui.Event`. It has the following properties:
* `event.action`, which could be `ti.ui.PRESS`, `ti.ui.RELEASE`, ...
* `event.key`, which indicates the key related to this event

To obtain the mouse position:
* `window.get_cursor_pos()`

To check if a specific key is currently pressed:
* `window.is_pressed(key)`



Here is an input processing example in GGUI version [`mpm128`](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/ggui_examples/mpm128_ggui.py):

```python
while window.running:
    # keyboard event processing
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r': reset()
        elif window.event.key in [ti.ui.ESCAPE]: break
    if window.event is not None: gravity[None] = [0, 0]  # if had any event
    if window.is_pressed(ti.ui.LEFT, 'a'): gravity[None][0] = -1
    if window.is_pressed(ti.ui.RIGHT, 'd'): gravity[None][0] = 1
    if window.is_pressed(ti.ui.UP, 'w'): gravity[None][1] = 1
    if window.is_pressed(ti.ui.DOWN, 's'): gravity[None][1] = -1

    # mouse event processing
    mouse = window.get_cursor_pos()
    # ...
    if window.is_pressed(ti.ui.LMB):
        attractor_strength[None] = 1
    if window.is_pressed(ti.ui.RMB):
        attractor_strength[None] = -1
```


## Image I/O

To write the current screen content into an image file:

```python
window.write_image(filename)
```

Notice that, when the window is showing, you have to call `window.write_image()` before the `window.show()` call.


## Off-screen rendering

GGUI supports rendering contents off-screen, that is, writing the results into image files without showing the window at all. This is sometimes referred to as "headless" rendering. To enable this mode, initialize the window with the argument `show_window=False`:

```python
window = ti.ui.Window('Window Title', (640, 360), show_window = False)
```
Then, you can use `window.write_image()` as normal, and remove the `window.show()` call at the end.
