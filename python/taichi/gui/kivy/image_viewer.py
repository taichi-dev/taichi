import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from builtins import (bytes, str, open, super, range, zip, round, input, int,
                      pow, object)

import taichi as tc
import numpy as np
import math


class Controllers(GridLayout):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.cols = 2
    self.update = None
    controllers = kwargs['controllers']

    def create_value_callback(lbl, name):

      def value_callback(_, value):
        self.__setattr__('value_' + name, value)
        lbl.text = name + ': ' + str(int(value))
        self.on_update()

      return value_callback

    for controller in controllers:
      lbl = Label(text=controller['name'] + ': ' + str(controller['value']))
      self.add_widget(lbl)
      cont = Slider(**controller)
      name = controller['name']
      self.add_widget(cont)
      self.__setattr__('value_' + name, controller['value'])
      cont.bind(value=create_value_callback(lbl, name))

  def on_update(self):
    if self.update is not None:
      self.update()


class ImageViewerWidget(GridLayout):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.cols = 2

    self.raw_img = tc.util.imread(
        tc.settings.get_asset_path('textures/vegas.jpg'))

    self.image = Image()
    controllers_info = [
        {
            'name': 'exposure',
            'min': 0,
            'max': 100,
            'value': 50,
        },
        {
            'name': 'temperature',
            'min': 0,
            'max': 100,
            'value': 50,
        },
        {
            'name': 'tint',
            'min': 0,
            'max': 100,
            'value': 50,
        },
        {
            'name': 'blur',
            'min': 0,
            'max': 10,
            'value': 0,
        },
    ]
    self.controllers = Controllers(controllers=controllers_info)
    self.controllers.update = lambda: self.update()
    self.add_widget(self.controllers)
    self.add_widget(self.image)
    self.update()

  def update(self, dt=0):
    img = self.raw_img * math.exp((self.controllers.value_exposure - 50) / 20)
    color_cast = np.array(
        ((1, math.exp(-(self.controllers.value_tint - 50) / 30), math.exp(
            -(self.controllers.value_temperature - 50) / 30))))
    color_cast *= math.sqrt(3) / np.linalg.norm(color_cast)
    img *= color_cast
    img = gaussian_blur(img, self.controllers.value_blur)
    tex = Texture.create(size=img.shape[:2])
    tex.blit_buffer(
        img.swapaxes(0, 1)[:, ::-1].flatten(),
        colorfmt='rgb',
        bufferfmt='float')

    self.image.texture = tex


def gaussian_blur(img, sigma):
  img = img.copy()
  #TODO: This is slow... We should directly blur Array2D<Vector3>
  for i in range(3):
    ch = img[:, :, i]
    ch = tc.util.ndarray_to_array2d(ch)
    ch = tc.core.gaussian_blur_2d_real(ch, sigma)
    ch = tc.util.array2d_to_ndarray(ch)
    img[:, :, i] = ch
  return img


class ImageViewerApp(App):

  def build(self):
    viewer = ImageViewerWidget()
    # Clock.schedule_interval(viewer.update, 1.0 / 60.0)
    return viewer


def show_image():
  ImageViewerApp().run()


show_image()
