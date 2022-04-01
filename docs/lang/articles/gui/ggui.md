---
sidebar_position: 2
---

# A New UI system: GGUI

| **Category** | **Prerequisites**                                            |
| ------------ | ------------------------------------------------------------ |
| OS           | Windows / Linux / Mac OS X                                   |
| Backend      | x64 / CUDA                                                   |
| Vulkan       | Install [the Vulkan environment](https://vulkan.lunarg.com/sdk/home). |
| LLVM         | 10.0.0 (Taichi customized version)                           |

Starting from v0.8.0, Taichi adds a new UI system GGUI. The new system uses GPU for rendering, making it much faster to render 3D scenes. That is why this new system gets its name as GGUI. This document describes the APIs that it provides.

:::note

It is recommended that you familiarize yourself with GGUI through the examples in `examples/ggui_examples`.

:::

## Create a window

`ti.ui.Window(name, res)` creates a window.

```python
window = ti.ui.Window('Window Title', (640, 360))
```

The following three types of objects can be displayed on a `ti.ui.Window`:

- 2D Canvas, which can be used to draw simple 2D geometries such as circles and triangles.

- 3D Scene, which can be used to render 3D meshes and particles, with a configurable camera and light sources.

- Immediate mode GUI components, for example buttons and textboxes.

## 2D Canvas

### Create a canvas

The following code retrieves a `Canvas` object that covers the entire window.

```python
canvas = window.get_canvas()
```

### Draw on the canvas

```python
canvas.set_background_color(color)
canvas.triangles(vertices, color, indices, per_vertex_color)
canvas.circles(vertices, radius, color, per_vertex_color)
canvas.lines(vertices, width, indices, color, per_vertex_color)
canvas.set_image(image)
```

The arguments `vertices`, `indices`, `per_vertex_color`, and `image` must be Taichi fields. If `per_vertex_color` is provided, `color` is ignored.

The positions/centers of geometries are represented as floats between `0.0` and `1.0`, which indicate the relative positions of the geometries on the canvas. For `circles()` and `lines()`, the `radius` and `width` arguments are relative to the height of the window.

The canvas is cleared after every frame. Always call these methods within the render loop.

## 3D Scene

### Create a scene

```python
scene = ti.ui.Scene()
```

### Configure camera

```python
camera = ti.ui.make_camera()
camera.position(pos)
camera.lookat(pos)
camera.up(dir)
camera.projection_mode(mode)
scene.set_camera(camera)
```

### Configuring light sources

#### Add a point light

Call `point_light()` to add a point light to the scene.

```python
scene.point_light(pos, color)
```

Note that you need to call `point_light()` for every frame. Similar to the `canvas()` methods, call this method within your render loop.

### 3D Geometries

```python
scene.mesh(vertices, indices, normals, color, per_vertex_color)
scene.particles(vertices, radius, color, per_vertex_color)
```

The arguments `vertices`, `indices`, `per_vertex_color`, and `image` are all expected to be Taichi fields. If `per_vertex_color` is provided, `color` is ignored.

The positions/centers of geometries should be in the world-space coordinates.

:::note

If a mesh has `num` triangles, the `indices` should be a 1D scalar field with a shape of `num * 3`, *not* a vector field.

`normals` is an optional parameter for `scene.mesh()`.

:::

### Rendering the scene

You can render a scene on a canvas.

```python
canvas.scene(scene)
```

## GUI components

The design of GGUI's GUI components follows the [Dear ImGui](https://github.com/ocornut/imgui) APIs.

```python
window.GUI.begin(name, x, y, width, height)
window.GUI.text(text)
is_clicked = window.GUI.button(name)
new_value = window.GUI.slider_float(name, old_value, min_value, max_value)
new_color = window.GUI.color_edit_3(name, old_color)
window.GUI.end()
```

## Show a window

Call `show()` to show a window.

```python
...
window.show()
```

Call this method *only* at the end of the render loop for each frame.

## User input processing

To retrieve the events that have occurred since the last method call:

```python
events = window.get_events()
```

Each `event` in `events` is an instance of `ti.ui.Event`. It has the following properties:

- `event.action`, which can be `ti.ui.PRESS`, `ti.ui.RELEASE`, or `ti.ui.MOTION`.

- `event.key`: the key related to this event.

To retrieve the mouse position:

- `window.get_cursor_pos()`

To check if a key is pressed:

- `window.is_pressed(key)`

The following is a user input processing example from **mpm128**:

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

To write the current frame in the window to an image file:

```python
window.write_image(filename)
```

Note that you must call `window.write_image()` before calling `window.show()`.

## Off-screen rendering

GGUI supports saving frames to images without showing the window. This is also known as "headless" rendering. To enable this mode, set the argument `show_window` to `False` when initializing a window.

```python
window = ti.ui.Window('Window Title', (640, 360), show_window = False)
```

Then you can call `window.write_image()` as normal and remove the `window.show()` call at the end.
