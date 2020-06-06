Export Your Results
===================
Taichi has some functions that can help you to **export results to images or videos**. This tutorial will tell you how to do that step by step.
Please make sure that your program runs well before reading this tutorial.

Export Images
-------------

- There are 2 common ways to export the result of your program (presumably a Taichi tensor) to images. The first one working with ti.GUI that it is more friendly to begginers. 
- The second approach is to call some built-in functions in Taichi to make things done, which is more flexible and harder.

Export Images by ti.GUI
+++++++++++++++++++++++

- For beginners and debug purposes, we recommend this first method by ti.GUI! That is to use **ti.GUI.show(filename)**. It can not only display the result to the screen but also export the result to your specified `filename`.
- Note that **the format of image is fully determined by your suffix**. 
- Though Taichi now supports exporting to .png, .jpg and .bmp these 3 mainstream formats, we strongly **recommend using .png** which has been tested well at this moment. For example:

.. code-block:: python
    
    import taichi as ti
    import os

    ti.init()

    pixel = ti.var(ti.u8, shape=(512, 512, 3))

    @ti.kernel
    def paint():
        for i, j, k in pixel:
            pixel[i, j, k] = ti.random() * 255

    iterations = 1000
    gui = ti.GUI("Export Result", res = 512)

    # mainloop
    for i in range(iterations):
        paint()
        gui.set_image(pixel)

        img_name = f"frame_{i}.png"   # create filename with suffix png
        print("Frame %d is recorded in %s" % (i, img_name))
        gui.show(img_name)  # export and show in GUI

- After running these codes above, you should get a batch of images in your "./results" folder.

Export Images by Taichi built-ins
+++++++++++++++++++++++++++++++++

- Why we have to instantiate GUI just for the sake of export images? The former method under the help of "ti.GUI" looks weird.
- If you are familiared with Taichi, we recommend using these following methods for manipulating images.
- This second approach is to use **ti.imwrite**, which is much more flexible compared with its former. For example:
    
    .. code-block:: python

        import taichi as ti

        ti.init()

        pixel = ti.var(ti.u8, shape=(512, 512, 3))

        @ti.kernel
        def set_pixel():
            for i, j, k in pixel:
                pixel[i, j, k] = ti.random() * 255

        set_pixel()
        img_name = f"imwrite_export.png"
        ti.imwrite(pixel.to_numpy(), img_name)
        print(f"The image has been saved to {img_name}")

- Taichi offers some helper functions that can read/write/show images easily. Please check :ref:`gui` for more details.

Export Videos
-------------

.. note::
    
    Taichi utils for exporting videos are dependent on ffmpeg. If ffmpeg hasn't been installed properly on your device, please follow the installation instructions of ffmpeg at the end of this document.

- **ti.VideoManger** can help you to export results in .mp4 or .git format. For example

.. code-block:: python

    import taichi as ti

    ti.init()

    pixel = ti.var(ti.u8, shape=(512, 512, 3))

    @ti.kernel
    def paint():
        for i, j, k in pixel:
            pixel[i, j, k] = ti.random() * 255

    result_dir = "./results"
    video_manger = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    for i in range(50):
        paint()

        pixel_img = pixel.to_numpy()
        video_manger.write_frame(pixel_img)
        print(f"\rFrame {i+1}/50 is recorded", end='')

    print("\nBegin to export .mp4 and .gif videos ...")
    video_manger.make_video(gif = True, mp4 = True)
    print("Mp4 video is saved to %s" % video_manger.get_output_filename(".mp4"))
    print("Gif video is saved to %s" % video_manger.get_output_filename(".gif"))
    
Running these codes above, you can find the resulting video in "./results/" folder :)

Install ffmpeg
--------------

Install ffmpeg on Windows
+++++++++++++++++++++++++

- Download the ffmpeg archive(named ffmpeg-2020xxx.zip) from `ffmpeg <https://ffmpeg.org/download.html>`_

- unzip this archive to where you would like, such as "D:/YOUR_FFMPEG_FOLDER"

- IMPORTANT: **add "D:/YOUR_FFMPEG_FOLDER/bin" to the environment variables**

- open the Windows CLI and type this line of code below to test the installation. The version info should be output if ffmpeg is set up properly.

.. code-block:: shell

    FFmpeg -version

Install FFmpeg on Linux
+++++++++++++++++++++++
- Most Linux distribution came with ``ffmpeg`` natively.
- Install ffmpeg on Ubuntu

.. code-block:: shell

    sudo apt-get update
    sudo apt-get install ffmpeg

- Install ffmpeg on CenteOS and RHEL

.. code-block:: shell

    sudo yum install ffmpeg ffmpeg-devel

- Install ffmpeg on Arch Linux:

.. code-block: shell

    sudo pacman -S ffmpeg

- test your installation by 

.. code-block:: shell

    ffmpeg -h

Install ffmpeg on OSX
+++++++++++++++++++++

- ffmpeg can be installed on OSX by brew

.. code-block:: shell

    brew install ffmpeg
