import sys
import tkinter as tk

from PIL import Image
from PIL.ImageTk import PhotoImage

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


class ImageViewer(object):

  def __init__(self, title, img):
    self.title = title

    self.root = get_top_level()

    self.root.configure(background='black')
    self.root.title(title)

    label = tk.Label(self.root, background='black')
    #label.pack()
    self.label = label

    self.root.lift()

    self.update(img)

  def update(self, img):
    img = Image.fromarray(img.swapaxes(0, 1)[::-1])
    photo = PhotoImage(img)
    self.label.image = photo
    self.label.configure(image=photo)
    self.label.pack()

  def callback(self):
    self.root.quit()
