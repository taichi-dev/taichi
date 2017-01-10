import os
import platform
import sys
import Tkinter as tk
import taichi as tc

from PIL import Image, ImageTk

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

tk_root = None
import numpy as np


def get_top_level():
    global tk_root
    if tk_root is None:
        tk_root = tk.Tk()
        return tk_root
    else:
        return tk.Toplevel()


def update_tk():
    assert tk_root is not None
    tk_root.update_idletasks()
    tk_root.update()


class EventHandler(FileSystemEventHandler):
    def __init__(self, filename, label, *args, **kwargs):
        self.filename = filename
        self.label = label

        super(EventHandler, self).__init__(*args, **kwargs)

    def on_created(self, event):
        if event.src_path != self.filename:
            return

        photo = ImageTk.PhotoImage(Image.open(self.filename))

        self.label.configure(image=photo)
        self.label.image = photo


class ImageWatchdog(object):
    def __init__(self, filename):
        self.filename = filename

    def callback(self):
        self.root.quit()

    def run(self):
        self.root = tk.Tk()

        self.root.configure(background='black')
        self.root.title('Frame Watchdog')
        self.root.protocol('WM_DELETE_WINDOW', self.callback)

        photo = ImageTk.PhotoImage(Image.open(self.filename))

        label = tk.Label(self.root, image=photo, background='black')
        label.image = photo  # keep a reference!
        label.pack()

        observer = Observer()
        event_handler = EventHandler(self.filename, label)

        observer.schedule(event_handler, os.path.dirname(self.filename))
        observer.start()

        # bring the window to the front when launched
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        self.root.mainloop()


class ImageViewer(object):
    def __init__(self, title, img):
        self.title = title

        self.root = get_top_level()

        self.root.configure(background='black')
        self.root.title(title)
        self.root.protocol('WM_DELETE_WINDOW', self.callback)

        label = tk.Label(self.root, background='black')
        label.pack()
        self.label = label

        # bring the window to the front when launched
        # it seems that the following code works well on my Mac,
        # so maybe we can avoid importing 'Cocoa'
        # What's your consideration here? @beaugunderson

        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        self.update(img)

    def update(self, img):
        photo = ImageTk.PhotoImage(Image.fromarray(img.swapaxes(0, 1)[::-1]))
        self.label.configure(image=photo)
        self.label.image = photo

    def callback(self):
        self.root.quit()


if __name__ == '__main__':
    app = ImageWatchdog(sys.argv[1])
    app.run()
