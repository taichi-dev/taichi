import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import threading
import time

def QImage_from_numpy_array(image):
  assert len(image.shape) == 3
  assert image.shape[2] == 3
  image = image.swapaxes(0, 1)[::-1].copy()
  if image.dtype in [np.float32, np.float64]:
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
  else:
    assert image.dtype == np.uint8, "image.dtype should be float32/64 or uint8"
  return QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)

class QtImageViewer(QWidget):
  def __init__(self, name, image):
    super().__init__()
    self.title = name
    self.setWindowTitle(self.title)
    label = QLabel(self)
    self.label = label
    self.update(image)
    
  def update(self, image):
    pixmap = QPixmap(QImage_from_numpy_array(image))
    self.label.setPixmap(pixmap)
    self.resize(pixmap.width(), pixmap.height())
    self.show()
    
viewers = {}
app = None

def create_window(name, img):
  if name not in viewers:
    signal = {}
    signal[0] = False
    def task():
      global app
      if app is None:
        app_ = QApplication(sys.argv)
      ex = QtImageViewer(name=name, image=img)
      signal[0] = True
      viewers[name] = ex
      if app is None:
        app = app_
        # app.exec_()
      while True:
        app.processEvents()
        ex.show()
        time.sleep(0.01)
    th = threading.Thread(target=task)
    th.start()
    while not signal[0]:
      pass
  else:
    viewers[name].update(img)
  

if __name__ == '__main__':
  test_image = np.zeros(shape=(500, 300, 3))
  for i in range(500):
    test_image[i] = 1
    create_window('Test A', test_image)
    create_window('Test B', test_image)
    create_window('Test C', test_image)

