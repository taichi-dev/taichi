# taichi.misc package

## Submodules

## taichi.misc.error module

## taichi.misc.gui module


### class taichi.misc.gui.GUI(name='Taichi', res=512, background_color=0, show_gui=True, fullscreen=False, fast_gui=False)
Bases: `object`

Taichi Graphical User Interface class.


* **Parameters**

    
    * **name** (*str**, **optional*) – The name of the GUI to be constructed.
    Default is ‘Taichi’.


    * **res** (*Union**[**int**, **List**[**int**]**]**, **optional*) – The resolution of created
    GUI. Default is 512\*512.


    * **background_color** (*int**, **optional*) – The background color of creted GUI.
    Default is 0x000000.


    * **show_gui** (*bool**, **optional*) – Specify whether to render the GUI. Default is True.


    * **fullscreen** (*bool**, **optional*) – Specify whether to render the GUI in
    fullscreen mode. Default is False.


    * **fast_gui** (*bool**, **optional*) – Specify whether to use fast gui mode of
    Taichi. Default is False.



* **Returns**

    The created taichi GUI object.



* **Return type**

    `GUI`



#### arrow(orig, dir, radius=1, color=16777215, \*\*kwargs)
Draw a single arrow on canvas.


* **Parameters**

    
    * **orig** (*List**[**Number**]*) – The position where arrow starts. Shape must be 2.


    * **dir** (*List**[**Number**]*) – The direction where arrow points to. Shape must be 2.


    * **radius** (*Number**, **optional*) – The width of arrow. Default is 1.


    * **color** (*int**, **optional*) – The color of arrow. Default is 0xFFFFFF.



#### arrow_field(dir, radius=1, color=16777215, bound=0.5, \*\*kwargs)
Draw a field of arrows on canvas.


* **Parameters**

    
    * **dir** (*np.array*) – The pattern and direction of the field of arrows.


    * **color** (*Union**[**int**, **np.array**]**, **optional*) – The color or colors of arrows.
    Default is 0xFFFFFF.


    * **bound** (*Number**, **optional*) – The boundary of the field. Default is 0.5.



#### arrows(orig, dir, radius=1, color=16777215, \*\*kwargs)
Draw a list arrows on canvas.


* **Parameters**

    
    * **orig** (*numpy.array*) – The positions where arrows start.


    * **dir** (*numpy.array*) – The directions where arrows point to.


    * **radius** (*Union**[**Number**, **np.array**]**, **optional*) – The width of arrows. Default is 1.


    * **color** (*Union**[**int**, **np.array**]**, **optional*) – The color or colors of arrows. Default is 0xffffff.



#### button(text, event_name=None)
Create a button object on canvas to be manipulated with.


* **Parameters**

    
    * **text** (*str*) – The title of button.


    * **event_name** (*str**, **optional*) – The event name associated with button.
    Default is WidgetButton_{text}



* **Returns**

    The event name associated with created button.



#### circle(pos, color=16777215, radius=1)
Draw a single circle on canvas.


* **Parameters**

    
    * **pos** (*Union**[**List**[**int**]**, **numpy.array**]*) – The position of the circle.


    * **color** (*int**, **Optional*) – The color of the circle. Default is 0xFFFFFF.


    * **radius** (*Number**, **Optional*) – The radius of the circle. Default is 1.



#### circles(pos, radius=1, color=16777215, palette=None, palette_indices=None)
Draw a list of circles on canvas.


* **Parameters**

    
    * **pos** (*numpy.array*) – The positions of the circles.


    * **radius** (*Number**, **optional*) – The radius of the circles. Default is 1.


    * **color** (*int**, **optional*) – The color of the circles. Default is 0xFFFFFF.


    * **palette** (*list**[**int**]**, **optional*) – The List of colors from which to
    choose to draw. Default is None.


    * **palette_indices** (*Union**[**list**[**int**]**, **ti.field**, **numpy.array**]**, **optional*) – The List of indices that choose color from palette for each
    circle. Shape must match pos. Default is None.



#### clear(color=None)
Clear the canvas with the color provided.


* **Parameters**

    **color** (*int**, **optional*) – Specify the color to clear the canvas. Default
    is the background color of GUI.



#### property fps_limit()
Get the property of fps limit.


* **Returns**

    The property of fps limit of gui.



#### get_cursor_pos()
Get the current position of mouse.


* **Returns**

    The current position of mouse.



#### get_event(\*filter)
Check if the specific event is triggered.


* **Parameters**

    **\*filter** (*ti.GUI.EVENT*) – The specific event to be checked.



* **Returns**

    Bool to indicate whether the specific event is triggered.



#### get_events(\*filter)
Get a list of events that are triggered.


* **Parameters**

    **\*filter** (*List**[**ti.GUI.EVENT**]*) – The type of events to be filtered.



* **Returns**

    A list of events that are triggered.



* **Return type**

    `EVENT`



#### get_image()
Get the image data.


* **Returns**

    The image data in numpy contiguous array type.



* **Return type**

    `numpy.array`



#### get_key_event()
Get keyboard triggered event.


* **Returns**

    The keyboard triggered event.



* **Return type**

    `EVENT`



#### has_key_event()
Check if there are any key event registered.


* **Returns**

    Bool to indicate whether there is any key event registered.



#### is_pressed(\*keys)
Check if the specific key or keys are pressed.


* **Parameters**

    **\*keys** (*Union**[**str**, **List**[**str**]**]*) – The string that stands for keys in keyboard.



* **Returns**

    Bool to indicate whether the key or keys are pressed.



#### label(text)
Create a label object on canvas.


* **Parameters**

    **text** (*str*) – The title of label.



* **Returns**

    The created label object.



* **Return type**

    `WidgetValue`



#### line(begin, end, radius=1, color=16777215)
Draw a single line on canvas.


* **Parameters**

    
    * **begin** (*List**[**Number**]*) – The position of one end of line. Shape must be 2.


    * **end** (*List**[**Number**]*) – The position of the other end of line. Shape must be 2.


    * **radius** (*Number**, **optional*) – The width of line. Default is 1.


    * **color** (*int**, **optional*) – The color of line. Default is 0xFFFFFF.



#### lines(begin, end, radius=1, color=16777215)
Draw a list of lines on canvas.


* **Parameters**

    
    * **begin** (*numpy.array*) – The positions of one end of lines.


    * **end** (*numpy.array*) – The positions of the other end of lines.


    * **radius** (*Union**[**Number**, **numpy.array**]**, **optional*) – The width of lines.
    Can be either a single width or a list of width whose shape matches
    the shape of begin & end. Default is 1.


    * **color** (*Union**[**int**, **numpy.array**]**, **optional*) – The color or colors of lines.
    Can be either a single color or a list of colors whose shape matches
    the shape of begin & end. Default is 0xFFFFFF.



#### point_field(radius, color=16777215, bound=0.5)
Draw a field of points on canvas.


* **Parameters**

    
    * **radius** (*np.array*) – The pattern and radius of the field of points.


    * **color** (*Union**[**int**, **np.array**]**, **optional*) – The color or colors of points.
    Default is 0xFFFFFF.


    * **bound** (*Number**, **optional*) – The boundary of the field. Default is 0.5.



#### rect(topleft, bottomright, radius=1, color=16777215)
Draw a single rectangle on canvas.


* **Parameters**

    
    * **topleft** (*List**[**Number**]*) – The position of the topleft corner of rectangle.
    Shape must be 2.


    * **bottomright** (*List**[**Number**]*) – The position of the bottomright corner
    of rectangle. Shape must be 2.


    * **radius** (*Number**, **optional*) – The width of rectangle’s sides. Default is 1.


    * **color** (*int**, **optional*) – The color of rectangle. Default is 0xFFFFFF.



#### property running()
Get the property of whether the gui is running.


* **Returns**

    The running property of gui(bool).



#### set_image(img)
Draw an image on canvas.


* **Parameters**

    **img** (*Union**[**ti.field**, **numpy.array**]*) – The color array representing the
    image to be drawn. Support greyscale, RG, RGB, and RGBA color
    representations. Its shape must match GUI resolution.



#### show(file=None)
Show the frame or save current frame as a picture.


* **Parameters**

    **file** (*str**, **optional*) – The path & name of the picture to be saved.
    Default is None.



#### slider(text, minimum, maximum, step=1)
Create a slider object on canvas to be manipulated with.


* **Parameters**

    
    * **text** (*str*) – The title of slider.


    * **minimum** (*Number*) – The minimum value of slider.


    * **maximum** (*Number*) – The maximum value of slider.


    * **step** (*Number**, **optional*) – The changing step of slider. Default is 1.



* **Returns**

    The created slider object.



* **Return type**

    `WidgetValue`



#### text(content, pos, font_size=15, color=16777215)
Draw texts on canvas.


* **Parameters**

    
    * **content** (*str*) – The text to be drawn on canvas.


    * **pos** (*List**[**Number**]*) – The position where the text is to be put.


    * **font_size** (*Number**, **optional*) – The font size of the text.


    * **color** (*int**, **optional*) – The color of the text. Default is 0xFFFFFF.



#### triangle(a, b, c, color=16777215)
Draw a single triangle on canvas.


* **Parameters**

    
    * **a** (*List**[**Number**]*) – The position of the first point of triangle. Shape must be 2.


    * **b** (*List**[**Number**]*) – The position of the second point of triangle. Shape must be 2.


    * **c** (*List**[**Number**]*) – The position of the third point of triangle. Shape must be 2.


    * **color** (*int**, **optional*) – The color of the triangle. Default is 0xFFFFFF.



#### triangles(a, b, c, color=16777215)
Draw a list of triangles on canvas.


* **Parameters**

    
    * **a** (*numpy.array*) – The positions of the first points of triangles.


    * **b** (*numpy.array*) – The positions of the second points of triangles.


    * **c** (*numpy.array*) – The positions of the thrid points of triangles.


    * **color** (*Union**[**int**, **numpy.array**]**, **optional*) – The color or colors of triangles.
    Can be either a single color or a list of colors whose shape matches
    the shape of a & b & c. Default is 0xFFFFFF.



### taichi.misc.gui.hex_to_rgb(color)
Convert hex color format to rgb color format.


* **Parameters**

    **color** (*int*) – The hex representation of color.



* **Returns**

    The rgb representation of color.



### taichi.misc.gui.rgb_to_hex(c)
Convert rgb color format to hex color format.


* **Parameters**

    **c** (*List**[**int**]*) – The rbg representation of color.



* **Returns**

    The hex representation of color.


## taichi.misc.image module


### taichi.misc.image.imdisplay(img)
Try to display image in interactive shell.


* **Parameters**

    **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of of array with shape (width, height) or (height, width, 3) or (height, width, 4).



### taichi.misc.image.imread(filename, channels=0)
Load image from a specific file.


* **Parameters**

    
    * **filename** (*str*) – An image filename to load from.


    * **channels** (*int**, **optinal*) – The channels hint of input image, Default to 0.



* **Returns**

    An output image loaded from given filename.



* **Return type**

    np.ndarray



### taichi.misc.image.imresize(img, w, h=None)
Resize an image to a specific size.


* **Parameters**

    
    * **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of of array with shape (width, height, …)


    * **w** (*int*) – The output width after resize.


    * **h** (*int**, **optional*) – The output height after resize, will be the same as width if not set. Default to None.



* **Returns**

    An output image after resize input.



* **Return type**

    np.ndarray



### taichi.misc.image.imshow(img, window_name='imshow')
Show image in a Taichi GUI.


* **Parameters**

    
    * **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of of array with shape (width, height) or (height, width, 3) or (height, width, 4).


    * **window_name** (*str**, **optional*) – The title of GUI window. Default to imshow.



### taichi.misc.image.imwrite(img, filename)
Save a field to a a specific file.


* **Parameters**

    
    * **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of shape (height, width) or (height, width, 3) or (height, width, 4),             if dtype is float-type (ti.f16, ti.f32, np.float32 etc), **the value of each pixel should be float between [0.0, 1.0]**. Otherwise ti.imwrite will first clip them into [0.0, 1.0]                if dtype is int-type (ti.u8, ti.u16, np.uint8 etc), , **the value of each pixel can be any valid integer in its own bounds**. These integers in this field will be scaled to [0, 255] by being divided over the upper bound of its basic type accordingly.


    * **filename** (*str*) – The filename to save to.


## taichi.misc.task module

## taichi.misc.util module


### taichi.misc.util.clear_profile_info()
Clear profiler’s records about time elapsed on the host tasks.

Call function imports from C++ : _ti_core.clear_profile_info()


### taichi.misc.util.deprecated(old, new, warning_type=<class 'DeprecationWarning'>)
Mark an API as deprecated.


* **Parameters**

    
    * **old** (*str*) – old method.


    * **new** (*str*) – new method.


    * **warning_type** (*builtin warning type*) – type of warning.


Example:

```
>>> @deprecated('ti.sqr(x)', 'x**2')
>>> def sqr(x):
>>>     return x**2
```


* **Returns**

    Decorated fuction with warning message



### taichi.misc.util.obsolete(old, new)
Mark an API as obsolete. Usage:

sqr = obsolete(‘ti.sqr(x)’, ‘x\*\*2’)


### taichi.misc.util.print_profile_info()
Print time elapsed on the host tasks in a hierarchical format.

This profiler is automatically on.

Call function imports from C++ : _ti_core.print_profile_info()

Example:

```
>>> import taichi as ti
>>> ti.init(arch=ti.cpu)
>>> var = ti.field(ti.f32, shape=1)
>>> @ti.kernel
>>> def compute():
>>>     var[0] = 1.0
>>>     print("Setting var[0] =", var[0])
>>> compute()
>>> ti.print_profile_info()
```

## Module contents


### class taichi.misc.GUI(name='Taichi', res=512, background_color=0, show_gui=True, fullscreen=False, fast_gui=False)
Bases: `object`

Taichi Graphical User Interface class.


* **Parameters**

    
    * **name** (*str**, **optional*) – The name of the GUI to be constructed.
    Default is ‘Taichi’.


    * **res** (*Union**[**int**, **List**[**int**]**]**, **optional*) – The resolution of created
    GUI. Default is 512\*512.


    * **background_color** (*int**, **optional*) – The background color of creted GUI.
    Default is 0x000000.


    * **show_gui** (*bool**, **optional*) – Specify whether to render the GUI. Default is True.


    * **fullscreen** (*bool**, **optional*) – Specify whether to render the GUI in
    fullscreen mode. Default is False.


    * **fast_gui** (*bool**, **optional*) – Specify whether to use fast gui mode of
    Taichi. Default is False.



* **Returns**

    The created taichi GUI object.



* **Return type**

    `GUI`



#### arrow(orig, dir, radius=1, color=16777215, \*\*kwargs)
Draw a single arrow on canvas.


* **Parameters**

    
    * **orig** (*List**[**Number**]*) – The position where arrow starts. Shape must be 2.


    * **dir** (*List**[**Number**]*) – The direction where arrow points to. Shape must be 2.


    * **radius** (*Number**, **optional*) – The width of arrow. Default is 1.


    * **color** (*int**, **optional*) – The color of arrow. Default is 0xFFFFFF.



#### arrow_field(dir, radius=1, color=16777215, bound=0.5, \*\*kwargs)
Draw a field of arrows on canvas.


* **Parameters**

    
    * **dir** (*np.array*) – The pattern and direction of the field of arrows.


    * **color** (*Union**[**int**, **np.array**]**, **optional*) – The color or colors of arrows.
    Default is 0xFFFFFF.


    * **bound** (*Number**, **optional*) – The boundary of the field. Default is 0.5.



#### arrows(orig, dir, radius=1, color=16777215, \*\*kwargs)
Draw a list arrows on canvas.


* **Parameters**

    
    * **orig** (*numpy.array*) – The positions where arrows start.


    * **dir** (*numpy.array*) – The directions where arrows point to.


    * **radius** (*Union**[**Number**, **np.array**]**, **optional*) – The width of arrows. Default is 1.


    * **color** (*Union**[**int**, **np.array**]**, **optional*) – The color or colors of arrows. Default is 0xffffff.



#### button(text, event_name=None)
Create a button object on canvas to be manipulated with.


* **Parameters**

    
    * **text** (*str*) – The title of button.


    * **event_name** (*str**, **optional*) – The event name associated with button.
    Default is WidgetButton_{text}



* **Returns**

    The event name associated with created button.



#### circle(pos, color=16777215, radius=1)
Draw a single circle on canvas.


* **Parameters**

    
    * **pos** (*Union**[**List**[**int**]**, **numpy.array**]*) – The position of the circle.


    * **color** (*int**, **Optional*) – The color of the circle. Default is 0xFFFFFF.


    * **radius** (*Number**, **Optional*) – The radius of the circle. Default is 1.



#### circles(pos, radius=1, color=16777215, palette=None, palette_indices=None)
Draw a list of circles on canvas.


* **Parameters**

    
    * **pos** (*numpy.array*) – The positions of the circles.


    * **radius** (*Number**, **optional*) – The radius of the circles. Default is 1.


    * **color** (*int**, **optional*) – The color of the circles. Default is 0xFFFFFF.


    * **palette** (*list**[**int**]**, **optional*) – The List of colors from which to
    choose to draw. Default is None.


    * **palette_indices** (*Union**[**list**[**int**]**, **ti.field**, **numpy.array**]**, **optional*) – The List of indices that choose color from palette for each
    circle. Shape must match pos. Default is None.



#### clear(color=None)
Clear the canvas with the color provided.


* **Parameters**

    **color** (*int**, **optional*) – Specify the color to clear the canvas. Default
    is the background color of GUI.



#### property fps_limit()
Get the property of fps limit.


* **Returns**

    The property of fps limit of gui.



#### get_cursor_pos()
Get the current position of mouse.


* **Returns**

    The current position of mouse.



#### get_event(\*filter)
Check if the specific event is triggered.


* **Parameters**

    **\*filter** (*ti.GUI.EVENT*) – The specific event to be checked.



* **Returns**

    Bool to indicate whether the specific event is triggered.



#### get_events(\*filter)
Get a list of events that are triggered.


* **Parameters**

    **\*filter** (*List**[**ti.GUI.EVENT**]*) – The type of events to be filtered.



* **Returns**

    A list of events that are triggered.



* **Return type**

    `EVENT`



#### get_image()
Get the image data.


* **Returns**

    The image data in numpy contiguous array type.



* **Return type**

    `numpy.array`



#### get_key_event()
Get keyboard triggered event.


* **Returns**

    The keyboard triggered event.



* **Return type**

    `EVENT`



#### has_key_event()
Check if there are any key event registered.


* **Returns**

    Bool to indicate whether there is any key event registered.



#### is_pressed(\*keys)
Check if the specific key or keys are pressed.


* **Parameters**

    **\*keys** (*Union**[**str**, **List**[**str**]**]*) – The string that stands for keys in keyboard.



* **Returns**

    Bool to indicate whether the key or keys are pressed.



#### label(text)
Create a label object on canvas.


* **Parameters**

    **text** (*str*) – The title of label.



* **Returns**

    The created label object.



* **Return type**

    `WidgetValue`



#### line(begin, end, radius=1, color=16777215)
Draw a single line on canvas.


* **Parameters**

    
    * **begin** (*List**[**Number**]*) – The position of one end of line. Shape must be 2.


    * **end** (*List**[**Number**]*) – The position of the other end of line. Shape must be 2.


    * **radius** (*Number**, **optional*) – The width of line. Default is 1.


    * **color** (*int**, **optional*) – The color of line. Default is 0xFFFFFF.



#### lines(begin, end, radius=1, color=16777215)
Draw a list of lines on canvas.


* **Parameters**

    
    * **begin** (*numpy.array*) – The positions of one end of lines.


    * **end** (*numpy.array*) – The positions of the other end of lines.


    * **radius** (*Union**[**Number**, **numpy.array**]**, **optional*) – The width of lines.
    Can be either a single width or a list of width whose shape matches
    the shape of begin & end. Default is 1.


    * **color** (*Union**[**int**, **numpy.array**]**, **optional*) – The color or colors of lines.
    Can be either a single color or a list of colors whose shape matches
    the shape of begin & end. Default is 0xFFFFFF.



#### point_field(radius, color=16777215, bound=0.5)
Draw a field of points on canvas.


* **Parameters**

    
    * **radius** (*np.array*) – The pattern and radius of the field of points.


    * **color** (*Union**[**int**, **np.array**]**, **optional*) – The color or colors of points.
    Default is 0xFFFFFF.


    * **bound** (*Number**, **optional*) – The boundary of the field. Default is 0.5.



#### rect(topleft, bottomright, radius=1, color=16777215)
Draw a single rectangle on canvas.


* **Parameters**

    
    * **topleft** (*List**[**Number**]*) – The position of the topleft corner of rectangle.
    Shape must be 2.


    * **bottomright** (*List**[**Number**]*) – The position of the bottomright corner
    of rectangle. Shape must be 2.


    * **radius** (*Number**, **optional*) – The width of rectangle’s sides. Default is 1.


    * **color** (*int**, **optional*) – The color of rectangle. Default is 0xFFFFFF.



#### property running()
Get the property of whether the gui is running.


* **Returns**

    The running property of gui(bool).



#### set_image(img)
Draw an image on canvas.


* **Parameters**

    **img** (*Union**[**ti.field**, **numpy.array**]*) – The color array representing the
    image to be drawn. Support greyscale, RG, RGB, and RGBA color
    representations. Its shape must match GUI resolution.



#### show(file=None)
Show the frame or save current frame as a picture.


* **Parameters**

    **file** (*str**, **optional*) – The path & name of the picture to be saved.
    Default is None.



#### slider(text, minimum, maximum, step=1)
Create a slider object on canvas to be manipulated with.


* **Parameters**

    
    * **text** (*str*) – The title of slider.


    * **minimum** (*Number*) – The minimum value of slider.


    * **maximum** (*Number*) – The maximum value of slider.


    * **step** (*Number**, **optional*) – The changing step of slider. Default is 1.



* **Returns**

    The created slider object.



* **Return type**

    `WidgetValue`



#### text(content, pos, font_size=15, color=16777215)
Draw texts on canvas.


* **Parameters**

    
    * **content** (*str*) – The text to be drawn on canvas.


    * **pos** (*List**[**Number**]*) – The position where the text is to be put.


    * **font_size** (*Number**, **optional*) – The font size of the text.


    * **color** (*int**, **optional*) – The color of the text. Default is 0xFFFFFF.



#### triangle(a, b, c, color=16777215)
Draw a single triangle on canvas.


* **Parameters**

    
    * **a** (*List**[**Number**]*) – The position of the first point of triangle. Shape must be 2.


    * **b** (*List**[**Number**]*) – The position of the second point of triangle. Shape must be 2.


    * **c** (*List**[**Number**]*) – The position of the third point of triangle. Shape must be 2.


    * **color** (*int**, **optional*) – The color of the triangle. Default is 0xFFFFFF.



#### triangles(a, b, c, color=16777215)
Draw a list of triangles on canvas.


* **Parameters**

    
    * **a** (*numpy.array*) – The positions of the first points of triangles.


    * **b** (*numpy.array*) – The positions of the second points of triangles.


    * **c** (*numpy.array*) – The positions of the thrid points of triangles.


    * **color** (*Union**[**int**, **numpy.array**]**, **optional*) – The color or colors of triangles.
    Can be either a single color or a list of colors whose shape matches
    the shape of a & b & c. Default is 0xFFFFFF.



### taichi.misc.clear_profile_info()
Clear profiler’s records about time elapsed on the host tasks.

Call function imports from C++ : _ti_core.clear_profile_info()


### taichi.misc.deprecated(old, new, warning_type=<class 'DeprecationWarning'>)
Mark an API as deprecated.


* **Parameters**

    
    * **old** (*str*) – old method.


    * **new** (*str*) – new method.


    * **warning_type** (*builtin warning type*) – type of warning.


Example:

```
>>> @deprecated('ti.sqr(x)', 'x**2')
>>> def sqr(x):
>>>     return x**2
```


* **Returns**

    Decorated fuction with warning message



### taichi.misc.hex_to_rgb(color)
Convert hex color format to rgb color format.


* **Parameters**

    **color** (*int*) – The hex representation of color.



* **Returns**

    The rgb representation of color.



### taichi.misc.imdisplay(img)
Try to display image in interactive shell.


* **Parameters**

    **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of of array with shape (width, height) or (height, width, 3) or (height, width, 4).



### taichi.misc.imread(filename, channels=0)
Load image from a specific file.


* **Parameters**

    
    * **filename** (*str*) – An image filename to load from.


    * **channels** (*int**, **optinal*) – The channels hint of input image, Default to 0.



* **Returns**

    An output image loaded from given filename.



* **Return type**

    np.ndarray



### taichi.misc.imresize(img, w, h=None)
Resize an image to a specific size.


* **Parameters**

    
    * **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of of array with shape (width, height, …)


    * **w** (*int*) – The output width after resize.


    * **h** (*int**, **optional*) – The output height after resize, will be the same as width if not set. Default to None.



* **Returns**

    An output image after resize input.



* **Return type**

    np.ndarray



### taichi.misc.imshow(img, window_name='imshow')
Show image in a Taichi GUI.


* **Parameters**

    
    * **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of of array with shape (width, height) or (height, width, 3) or (height, width, 4).


    * **window_name** (*str**, **optional*) – The title of GUI window. Default to imshow.



### taichi.misc.imwrite(img, filename)
Save a field to a a specific file.


* **Parameters**

    
    * **img** (*Union**[**ti.field**, **np.ndarray**]*) – A field of shape (height, width) or (height, width, 3) or (height, width, 4),             if dtype is float-type (ti.f16, ti.f32, np.float32 etc), **the value of each pixel should be float between [0.0, 1.0]**. Otherwise ti.imwrite will first clip them into [0.0, 1.0]                if dtype is int-type (ti.u8, ti.u16, np.uint8 etc), , **the value of each pixel can be any valid integer in its own bounds**. These integers in this field will be scaled to [0, 255] by being divided over the upper bound of its basic type accordingly.


    * **filename** (*str*) – The filename to save to.



### taichi.misc.obsolete(old, new)
Mark an API as obsolete. Usage:

sqr = obsolete(‘ti.sqr(x)’, ‘x\*\*2’)


### taichi.misc.print_profile_info()
Print time elapsed on the host tasks in a hierarchical format.

This profiler is automatically on.

Call function imports from C++ : _ti_core.print_profile_info()

Example:

```
>>> import taichi as ti
>>> ti.init(arch=ti.cpu)
>>> var = ti.field(ti.f32, shape=1)
>>> @ti.kernel
>>> def compute():
>>>     var[0] = 1.0
>>>     print("Setting var[0] =", var[0])
>>> compute()
>>> ti.print_profile_info()
```


### taichi.misc.rgb_to_hex(c)
Convert rgb color format to hex color format.


* **Parameters**

    **c** (*List**[**int**]*) – The rbg representation of color.



* **Returns**

    The hex representation of color.
