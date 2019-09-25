import os

for i in range(7, 512, 2):
  os.system('python3 diffmpm_renderer.py snow 0040 {}'.format(i))