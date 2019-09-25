import os

for i in range(63, 1023, 16):
  os.system('python3 diffmpm_renderer.py snow 0040 {}'.format(i))