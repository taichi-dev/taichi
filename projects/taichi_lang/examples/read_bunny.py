import numpy as np

volume = np.fromfile('bunny_128.bin', dtype=np.float32).reshape((128, 128, 128))
import cv2
for i in range(128):
  cv2.imshow('bunny', volume[i, :, :])
  cv2.waitKey(10)
