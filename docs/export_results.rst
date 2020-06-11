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

    pixels = ti.var(ti.u8, shape=(512, 512, 3))

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

        pixels = ti.var(ti.u8, shape=(512, 512, 3))

        @ti.kernel
        def set_pixels():
            for i, j, k in pixels:
                pixels[i, j, k] = ti.random() * 255

        set_pixels()
        filename = f'imwrite_export.png'
        ti.imwrite(pixels.to_numpy(), filename)
        print(f'The image has been saved to {filename}')

- ``ti.imwrite`` can export Taichi tensors (``ti.Matrix``, ``ti.Vector``, ``ti.var``) and numpy tensors ``np.ndarray``.
- Same as above ``ti.GUI.show(filename)``, the image format (``png``, ``jpg`` and ``bmp``) is also controlled by the suffix of ``filename`` in ``ti.imwrite(filename)``.
- Meanwhile, the resulted image type (grayscale, RGB, or RGBA) is determined by **the number of channels in the input tensor**, i.e., the length of the third dimension (``tensor.shape[2]``).
- In other words, a tensor that has shape ``(w, h)`` or ``(w, h, 1)`` will be exported as a grayscale image.
- If you want to export ``RGB`` or ``RGBA`` images instead, the input tensor should have a shape ``(w, h, 3)`` or ``(w, h, 4)`` respectively.

.. note::

    All Taichi tensors have their own data types, such as ``ti.u8`` and ``ti.f32``. Different data types can lead to different behaviors of ``ti.imwrite``. Please check out :ref:`gui` for more details.

- Taichi offers other helper functions that read and show images in addition to ``ti.imwrite``. They are also demonstrated in :ref:`gui`.

Export videos
-------------

.. note::

    The video export utilities of Taichi depend on ``ffmpeg``. If ``ffmpeg`` is not installed on your machine, please follow the installation instructions of ``ffmpeg`` at the end of this page.

- ``ti.VideoManger`` can help you export results in ``mp4`` or ``gif`` format. For example,

.. code-block:: python

    import taichi as ti

    ti.init()

    pixels = ti.var(ti.u8, shape=(512, 512, 3))

    @ti.kernel
    def paint():
        for i, j, k in pixels:
            pixels[i, j, k] = ti.random() * 255

    result_dir = "./results"
    video_manger = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    for i in range(50):
        paint()

        pixels_img = pixels.to_numpy()
        video_manger.write_frame(pixels_img)
        print(f'\rFrame {i+1}/50 is recorded', end='')

    print()
    print('Exporting .mp4 and .gif videos...')
    video_manger.make_video(gif=True, mp4=True)
    print(f'MP4 video is saved to {video_manger.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manger.get_output_filename(".gif")}')

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
