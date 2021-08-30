---
sidebar_position: 1

---

# A New UI system: GGUI

A new UI system is to be added to Taichi. The new GUI system will use GPU for rendering, which will enable it to be much faster and to render 3d scenes. For this reason, this new system is sometimes referred to as GGUI. This doc describes the APIs provided.

Apart from this doc, a good way of getting familiarized with GGUI is to look at the examples. Please checkout the example code provided in  `examples/ggui_examples`.

## Creating a window

`ti.ui.Window(name, res)` creates a window.

```python
window = ti.ui.Window('Window Title', (640, 360))
```

There're three types of objects that can be displayed on a `ti.ui.Window`:

* 2D Canvas, which can be used to draw simple 2D geometries such as circles, triangles, etc.
* 3D Scene, which can be used to render 3D meshes and particles, with a configurable camera and light sources.
* Immediate mode GUI components, buttons, textboxes, etc.

## 2D Canvas

### Creating a canvas

```python
canvas = window.get_canvas()
```
this retrieves a `Canvas` object that covers the entire window.

### Drawing on the canvas

```python
canvas.set_back_ground_color(color)
canvas.triangles(vertices,color,indices,per_vertex_color)
canvas.circles(vertices,radius,color,per_vertex_color)
canvas.lines(vertices,width,indices,color,per_vertex_color)
canvas.set_image(image)
```

The arguments `vertices`, `indices`, `per_vertex_color`, `image` are all expected to be `taichi` fields. If `per_vertex_color` is provided, `color` will be ignored.

The positions/centers of geometries will be represented as floats between 0 and 1, which indicate relative positions on the canvas. For `circles` and `lines`, the `radius` and `width` argument is relative to the height of the window.

The canvas is cleared after every frame. You should call these methods within the render loop.


## 3D Scene

### Creating a scene
```python
scene =  ti.ui.Scene()
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
#### adding a point light
```python
scene.point_light(pos,color)
```
Note that, `point_light` needs to be called every frame. Similar to the `canvas` methods, you should call this within your render loop.


### 3d Geometries
```python
scene.mesh(vertices,indices,normals,color,per_vertex_color)
scene.particles(vertices,radius,color,per_vertex_color)
```

Again, the arguments `vertices`, `indices`, `per_vertex_color`, `image` are all expected to be `taichi` fields. If `per_vertex_color` is provided, `color` will be ignored.

The positions/centers of geometries should be in world-space coordinates.


### Rendering the scene
A scene can be rendered on a canvas.
```python
canvas.scene(scene)
```

## GUI components

The support for GUI components will closely follow Dear IMGUI (and will likely be implemented using it..).

```python
window.GUI.begin(name,x,y,width,height)
window.GUI.text(text)
is_clicked = window.GUI.button(name)
new_value = window.GUI.slider_float(name,old_value,min_value,max_value)
new_color = window.GUI.slider_float(name,old_color)
window.GUI.end()
```


## Showing a window
```python
...
window.show()
```
Call this method at the very end of the frame

## User Input Processing
To obtain the events that have occurred since the previous poll:

```python
events = window.get_events()
```

Each `event` in `events` is an instance of `ti.ui.Event`. It has the following properties:
* `event.action`, which could be `ti.ui.PRESS`, `ti.ui.RELEASE`, ...
* `event.key`, which indicates the key related to this event

To obtain mouse position:
* `window.get_cursor_pos()`

To check if a specific key is currently pressed:
* `window.is_pressed(key)`
