import numpy as np
import os

for f in os.listdir('dataset'):
  print('Running on {}'.format(f))
  os.system('./vdb_baseline {}'.format(f))