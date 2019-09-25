import os

for i in range(7, 1024, 8):
  os.system('python3 diffmpm_renderer.py snow 0010 {}'.format(i))