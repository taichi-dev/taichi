from tk.viewer import ImageViewer, update_tk

viewers = {}

def show_image(name, img):
    if name in viewers:
        viewers[name].update(img)
    else:
        viewers[name] = ImageViewer(name, img)
    update_tk()



# TODO: destory viewers atexit
