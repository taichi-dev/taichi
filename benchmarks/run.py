import os

for f in sorted(os.listdir('.')):
  if f != 'run.py' and f.endswith('.py'):
    suite = None
    f = f[:-3]
    exec(f'import {f} as suite')
    benchmarks = list(sorted(filter(lambda x: x.startswith('benchmark_'), dir(suite))))
    print(f'Suite {f}:')
    for b in benchmarks:
      print(' *', b)

def run():
  return