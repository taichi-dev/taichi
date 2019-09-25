import os

for i in range(7, 512, 8):
  os.system('python3 diffmpm_renderer.py snow 0090 {}'.format(i))