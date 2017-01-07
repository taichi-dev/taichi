#!/usr/bin/env python

import os
import platform
import sys
import Tkinter as tk

from PIL import Image, ImageTk

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


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


class App(object):

    def __init__(self, filename):
        self.filename = filename

    def callback(self):
        self.root.quit()

    def run(self):
        self.root = tk.Tk()

        self.root.configure(background='black')
        self.root.title('Frame Viewer')
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
        if platform.system != 'Darwin':
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(self.root.attributes, '-topmost', False)
        else:
            from Cocoa import NSRunningApplication, NSApplicationActivateIgnoringOtherApps

            app = NSRunningApplication.runningApplicationWithProcessIdentifier(os.getpid())
            app.activateWithOptions(NSApplicationActivateIgnoringOtherApps)

        self.root.mainloop()


if __name__ == '__main__':
    app = App(sys.argv[1])
    app.run()
