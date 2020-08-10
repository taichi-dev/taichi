.. _export_results:

Export your results
===================
Taichi has functions that help you **export visual results to images or videos**. This tutorial demonstrates how to use them step by step.

Export images
-------------

- There are two ways to export visual results of your program to images.
- The first and easier way is to make use of ``ti.GUI``.
- The second way is to call some Taichi functions such as ``ti.imwrite``.

Export images using ``ti.GUI.show``
+++++++++++++++++++++++++++++++++++

- ``ti.GUI.show(filename)`` can not only display the GUI canvas on your screen, but also save the image to your specified ``filename``.
- Note that the format of the image is fully determined by the suffix of ``filename``.
- Taichi now supports saving to ``png``, ``jpg``, and ``bmp`` formats.
- We recommend using ``png`` format. For example:

.. code-block:: python

    import taichi as ti
    import os

    ti.init()

    pixels = ti.field(ti.u8, shape=(512, 512, 3))

    @ti.kernel
    def paint():
        for i, j, k in pixels:
            pixels[i, j, k] = ti.random() * 255

    iterations = 1000
    gui = ti.GUI("Random pixels", res=512)

    # mainloop
    for i in range(iterations):
        paint()
        gui.set_image(pixels)

        filename = f'frame_{i:05d}.png'   # create filename with suffix png
        print(f'Frame {i} is recorded in {filename}')
        gui.show(filename)  # export and show in GUI

- After running the code above, you will get a series of images in the current folder.

Export images using ``ti.imwrite``
++++++++++++++++++++++++++++++++++
To save images without invoking ``ti.GUI.show(filename)``, use ``ti.imwrite(filename)``. For example:

    .. code-block:: python

        import taichi as ti

        ti.init()

        pixels = ti.field(ti.u8, shape=(512, 512, 3))

        @ti.kernel
        def set_pixels():
            for i, j, k in pixels:
                pixels[i, j, k] = ti.random() * 255

        set_pixels()
        filename = f'imwrite_export.png'
        ti.imwrite(pixels.to_numpy(), filename)
        print(f'The image has been saved to {filename}')

- ``ti.imwrite`` can export Taichi fields (``ti.Matrix.field``, ``ti.Vector.field``, ``ti.field``) and numpy arrays ``np.ndarray``.
- Same as above ``ti.GUI.show(filename)``, the image format (``png``, ``jpg`` and ``bmp``) is also controlled by the suffix of ``filename`` in ``ti.imwrite(filename)``.
- Meanwhile, the resulted image type (grayscale, RGB, or RGBA) is determined by **the number of channels in the input field**, i.e., the length of the third dimension (``field.shape[2]``).
- In other words, a field that has shape ``(w, h)`` or ``(w, h, 1)`` will be exported as a grayscale image.
- If you want to export ``RGB`` or ``RGBA`` images instead, the input field should have a shape ``(w, h, 3)`` or ``(w, h, 4)`` respectively.

.. note::

    All Taichi fields have their own data types, such as ``ti.u8`` and ``ti.f32``. Different data types can lead to different behaviors of ``ti.imwrite``. Please check out :ref:`gui` for more details.

- Taichi offers other helper functions that read and show images in addition to ``ti.imwrite``. They are also demonstrated in :ref:`gui`.

Export videos
-------------

.. note::

    The video export utilities of Taichi depend on ``ffmpeg``. If ``ffmpeg`` is not installed on your machine, please follow the installation instructions of ``ffmpeg`` at the end of this page.

- ``ti.VideoManager`` can help you export results in ``mp4`` or ``gif`` format. For example,

.. code-block:: python

    import taichi as ti

    ti.init()

    pixels = ti.field(ti.u8, shape=(512, 512, 3))

    @ti.kernel
    def paint():
        for i, j, k in pixels:
            pixels[i, j, k] = ti.random() * 255

    result_dir = "./results"
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    for i in range(50):
        paint()

        pixels_img = pixels.to_numpy()
        video_manager.write_frame(pixels_img)
        print(f'\rFrame {i+1}/50 is recorded', end='')

    print()
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')

After running the code above, you will find the output videos in the ``./results/`` folder.

Install ffmpeg
--------------

Install ffmpeg on Windows
+++++++++++++++++++++++++

- Download the ``ffmpeg`` archive(named ``ffmpeg-2020xxx.zip``) from `ffmpeg <https://ffmpeg.org/download.html>`_;

- Unzip this archive to a folder, such as "D:/YOUR_FFMPEG_FOLDER";

- **Important:** add ``D:/YOUR_FFMPEG_FOLDER/bin`` to the ``PATH`` environment variable;

- Open the Windows ``cmd`` or ``PowerShell`` and type the line of code below to test your installation. If ``ffmpeg`` is set up properly, the version information will be printed.

.. code-block:: shell

    ffmpeg -version

Install ``ffmpeg`` on Linux
+++++++++++++++++++++++++++
- Most Linux distribution came with ``ffmpeg`` natively, so you do not need to read this part if the ``ffmpeg`` command is already there on your machine.
- Install ``ffmpeg`` on Ubuntu

.. code-block:: shell

    sudo apt-get update
    sudo apt-get install ffmpeg

- Install ``ffmpeg`` on CentOS and RHEL

.. code-block:: shell

    sudo yum install ffmpeg ffmpeg-devel

- Install ``ffmpeg`` on Arch Linux:

.. code-block: shell

    sudo pacman -S ffmpeg

- Test your installation using

.. code-block:: shell

    ffmpeg -h

Install ``ffmpeg`` on OS X
++++++++++++++++++++++++++

- ``ffmpeg`` can be installed on OS X using ``homebrew``:

.. code-block:: shell

    brew install ffmpeg

.. _export_ply_files:

Export PLY files
----------------
- ``ti.PLYwriter`` can help you export results in the ``ply`` format. Below is a short example of exporting 10 frames of a moving cube with vertices randomly colored,

.. code-block:: python

    import taichi as ti
    import numpy as np

    ti.init(arch=ti.cpu)

    num_vertices = 1000
    pos = ti.Vector.field(3, dtype=ti.f32, shape=(10, 10, 10))
    rgba = ti.Vector.field(4, dtype=ti.f32, shape=(10, 10, 10))


    @ti.kernel
    def place_pos():
        for i, j, k in pos:
            pos[i, j, k] = 0.1 * ti.Vector([i, j, k])


    @ti.kernel
    def move_particles():
        for i, j, k in pos:
            pos[i, j, k] += ti.Vector([0.1, 0.1, 0.1])


    @ti.kernel
    def fill_rgba():
        for i, j, k in rgba:
            rgba[i, j, k] = ti.Vector(
                [ti.random(), ti.random(), ti.random(), ti.random()])


    place_pos()
    series_prefix = "example.ply"
    for frame in range(10):
        move_particles()
        fill_rgba()
        # now adding each channel only supports passing individual np.array
        # so converting into np.ndarray, reshape
        # remember to use a temp var to store so you dont have to convert back
        np_pos = np.reshape(pos.to_numpy(), (num_vertices, 3))
        np_rgba = np.reshape(rgba.to_numpy(), (num_vertices, 4))
        # create a PLYWriter
        writer = ti.PLYWriter(num_vertices=num_vertices)
        writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        writer.add_vertex_rgba(
            np_rgba[:, 0], np_rgba[:, 1], np_rgba[:, 2], np_rgba[:, 3])
        writer.export_frame_ascii(frame, series_prefix)

After running the code above, you will find the output sequence of ``ply`` files in the current working directory. Next, we will break down the usage of ``ti.PLYWriter`` into 4 steps and show some examples.

- Setup ``ti.PLYWriter``

.. code-block:: python

    # num_vertices must be a positive int
    # num_faces is optional, default to 0
    # face_type can be either "tri" or "quad", default to "tri"

    # in our previous example, a writer with 1000 vertices and 0 triangle faces is created
    num_vertices = 1000
    writer = ti.PLYWriter(num_vertices=num_vertices)

    # in the below example, a writer with 20 vertices and 5 quadrangle faces is created
    writer2 = ti.PLYWriter(num_vertices=20, num_faces=5, face_type="quad")

- Add required channels

.. code-block:: python

    # A 2D grid with quad faces
    #     y
    #     |
    # z---/
    #    x
    #         19---15---11---07---03
    #         |    |    |    |    |
    #         18---14---10---06---02
    #         |    |    |    |    |
    #         17---13---19---05---01
    #         |    |    |    |    |
    #         16---12---08---04---00

    writer = ti.PLYWriter(num_vertices=20, num_faces=12, face_type="quad")

    # For the vertices, the only required channel is the position,
    # which can be added by passing 3 np.array x, y, z into the following function.

    x = np.zeros(20)
    y = np.array(list(np.arange(0, 4))*5)
    z = np.repeat(np.arange(5), 4)
    writer.add_vertex_pos(x, y, z)

    # For faces (if any), the only required channel is the list of vertex indices that each face contains.
    indices = np.array([0, 1, 5, 4]*12)+np.repeat(
        np.array(list(np.arange(0, 3))*4)+4*np.repeat(np.arange(4), 3), 4)
    writer.add_faces(indices)

- Add optional channels

.. code-block:: python

    # Add custome vertex channel, the input should include a key, a supported datatype and, the data np.array
    vdata = np.random.rand(20)
    writer.add_vertex_channel("vdata1", "double", vdata)

    # Add custome face channel
    foo_data = np.zeros(12)
    writer.add_face_channel("foo_key", "foo_data_type", foo_data)
    # error! because "foo_data_type" is not a supported datatype. Supported ones are
    # ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double']

    # PLYwriter already defines several useful helper functions for common channels
    # Add vertex color, alpha, and rgba
    # using float/double r g b alpha to reprent color, the range should be 0 to 1
    r = np.random.rand(20)
    g = np.random.rand(20)
    b = np.random.rand(20)
    alpha = np.random.rand(20)
    writer.add_vertex_color(r, g, b)
    writer.add_vertex_alpha(alpha)
    # equivilantly
    # add_vertex_rgba(r, g, b, alpha)

    # vertex normal
    writer.add_vertex_normal(np.ones(20), np.zeros(20), np.zeros(20))

    # vertex index, and piece (group id)
    writer.add_vertex_id()
    writer.add_vertex_piece(np.ones(20))

    # Add face index, and piece (group id)
    # Indexing the existing faces in the writer and add this channel to face channels
    writer.add_face_id()
    # Set all the faces is in group 1
    writer.add_face_piece(np.ones(12))

- Export files

.. code-block:: python

    series_prefix = "example.ply"
    series_prefix_ascii = "example_ascii.ply"
    # Export a single file
    # use ascii so you can read the content
    writer.export_ascii(series_prefix_ascii)

    # alternatively, use binary for a bit better performance
    # writer.export(series_prefix)

    # Export a sequence of files, ie in 10 frames
    for frame in range(10):
        # write each frame as i.e. "example_000000.ply" in your current running folder
        writer.export_frame_ascii(frame, series_prefix_ascii)
        # alternatively, use binary
        # writer.export_frame(frame, series_prefix)

        # update location/color
        x = x + 0.1*np.random.rand(20)
        y = y + 0.1*np.random.rand(20)
        z = z + 0.1*np.random.rand(20)
        r = np.random.rand(20)
        g = np.random.rand(20)
        b = np.random.rand(20)
        alpha = np.random.rand(20)
        # re-fill
        writer = ti.PLYWriter(num_vertices=20, num_faces=12, face_type="quad")
        writer.add_vertex_pos(x, y, z)
        writer.add_faces(indices)
        writer.add_vertex_channel("vdata1", "double", vdata)
        writer.add_vertex_color(r, g, b)
        writer.add_vertex_alpha(alpha)
        writer.add_vertex_normal(np.ones(20), np.zeros(20), np.zeros(20))
        writer.add_vertex_id()
        writer.add_vertex_piece(np.ones(20))
        writer.add_face_id()
        writer.add_face_piece(np.ones(12))

Import ``ply`` files into Houdini and Blender
+++++++++++++++++++++++++++++++++++++++++++++
Houdini supports importing a series of ``ply`` files sharing the same prefix/post-fix. Our ``export_frame`` can achieve the requirement for you. In Houdini, click ``File->Import->Geometry`` and navigate to the folder containing your frame results, who should be collapsed into one single entry like ``example_$F6.ply (0-9)``. Double-click this entry to finish the importing process.

Blender requires an add-on called `Stop-motion-OBJ <https://github.com/neverhood311/Stop-motion-OBJ>`_ to load the result sequences. `Detailed documentation <https://github.com/neverhood311/Stop-motion-OBJ/wiki>`_ is provided by the author on how to install and use the add-on. If you're using the latest version of Blender (2.80+), download and install the `latest release <https://github.com/neverhood311/Stop-motion-OBJ/releases/latest>`_ of Stop-motion-OBJ. For Blender 2.79 and older, use version ``v1.1.1`` of the add-on.
