import numpy as np
import os

os.makedirs('results', exist_ok=True)

for f in os.listdir('vdb_dataset'):
  print('Running on {}'.format(f))
  os.system('ti benchmark_vdb {}'.format(f))
