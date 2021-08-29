---
sidebar_position: 1

---

# New UI system

A new UI system is to be added to Taichi. The new GUI system will use GPU for rendering, which will enable it to be much faster and to render 3d scenes (for this reason, this new system is somtimes referred to as GGUI). This doc describes the APIs proveded.

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
canvas.set_back_ground_color(...)
canvas.triangles(...)
canvas.circles(...)
canvas.lines(...)
canvas.set_image(...)
```

To see examples of how

The positions/centers of geometries will be represented as floats between 0 and 1, which indicate relative positions on the canvas.



## 3D Scene

### Creating a scene
```python
scene =  ti.ui.Scene()
```
### Configuring camera
```python
camera = ti.ui.make_camera()
camera.lookat(pos)
camera.up(dir)
camera.center(pos)
camera.projection_mode(mode)
scene.set_camera(camera)
```
where `mode` is either `ti.ui.Scene.PROJECTION_PERSPECTIVE` or `ti.ui.Scene.PROJECTION_ORTHOGONAL`


### Configuring light sources
#### adding a light source
```python
scene.point_light(pos,color) 
```


### 3d Geometries
```python
scene.mesh(vertices,indices,color)
scene.particles(positions,radius,color)
```


### Rendering the scene 
a scene is rendered by first rendering it on a canvas.
```python
canvas.render(scene)
```

## GUI components

The support for GUI components will closely follow Dear IMGUI (and will likely be implemented using it..).

```python
window.GUI.begin(name,x,y,width,height)
window.GUI.text(...)
window.GUI.button(...)
window.GUI.end()
```


## Clearing and showing a window
```python
...
window.show()
```


## Events Processing
To obtain the events that have occurred since the previous poll:

```python
events = window.get_events()
```

Each `event` in `events` is an instance of `ti.ui.Event`. It has the following properties:
* `event.action`, which could be `ti.ui.ACTION_PRESSED`, `ti.ui.ACTION_RELEASED`, ...
* `event.key`, which is `ti.ui.KEY_XXXX`

To obtain mouse position:
* `window.get_mouse_position()`


## Example Application

```python
import taichi as ti

window = ti.ui.Window("Amazing Window",res)
canvas = window.get_canvas()
scene = ti.ui.Scene()


while window.running:
  events = window.get_event()
  if ev.action == ti.ui.ACTION_PRESSED and ev.key == ti.ui.KEY_SHIFT:
      ...

  canvas.clear(...)
  canvas.triangles(...)

  scene.clear()
  camera = ti.ui.make_camera()
  camera.lookat(pos)
  camera.up(dir)
  camera.center(pos)
  camera.projection_mode(mode)
  scene.set_camera(camera)
  scene.point_light(pos,color) 
  scene.mesh(...)
  canvas.render(scene)
  
  window.GUI.begin(name,x,y,width,height)
  window.GUI.text(...)
  window.GUI.button(...)
  window.GUI.end()

  window.show()
  
    
```