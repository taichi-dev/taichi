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
    pixel = ti.var(ti.u8, shape=(512, 512, 3))

ti.init()
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
- This second approach is to use **ti.imwrite**. It is much more flexible compared with its former.

.. function:: ti.imwrite(img, filename)

    :parameter img: (Matrix or Expr) the images you want to export
    :parameter filename: (string) filename you want to save

    Please make sure that the input image is a Taichi Matrix or Expr and **it has shape [height, width, compoents] correctly**. For example:
    
    .. code-block:: python

        import taichi as ti
        pixel = ti.var(ti.u8, shape=(512, 512, 3))

        @ti.kernel
        def set_pixel():
            for I in ti.grouped(pixel):
                pixel[I] = ti.random() * 255
        
        set_pixel()
        ti.imwrite(pixel.to_numpy(), f"imwrite_export.png")

.. function:: ti.imread(filename, channels = 0)

    :parameter filename: (string) filename of the image you want to load
    :parameter channels: (int) the number of channels in your specified image, the default value is zero.
    
    This function can load an image from the target filename, return it as a np.ndarray

.. function:: ti.imshow(img, windname)

    :parameter img: (Matrix or Expr) the image you want to show in the GUI
    :parameter windname: (string) the name of GUI window

    This function will open an instance of ti.GUI and show the input image on the screen.


Export Videos
-------------

.. note::
    
    Taichi utils for exporting videos are dependent on ffmpeg. If ffmpeg hasn't been installed properly on your device, please follow the installation instructions of ffmpeg at the end of this document.

- **ti.VideoManger** can help you to export results in .mp4 or .git format. For example

.. code-block:: python

    import taichi as ti

    pixel = ti.var(ti.u8, shape=(512, 512, 3))

    @ti.kernel
    def paint():
        for I in ti.grouped(pixel):
            pixel[I] = ti.random() * 255

    result_dir = "./results"
    video_manger = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    for i in range(50):
        paint()

        pixel_img = pixel.to_numpy()
        video_manger.write_frame(pixel_img)
        print("\rframe %d/%d" % (i, 50), end='')

    video_manger.make_video(gif = True, mp4 = True)
    print("mp4 video saved to %s" % video_manger.get_output_filename(".mp4"))
    print("gif video saved to %s" % video_manger.get_output_filename(".gif"))
    
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
