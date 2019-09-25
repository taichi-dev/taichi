import os

for i in range(7, 512, 32):
  os.system('python3 diffmpm_renderer.py snow 0020 {}'.format(i))