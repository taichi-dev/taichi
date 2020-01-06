import os

class Suite:
  def __init__(self, filename):
    self.cases = []
    self.name = filename[:-3]
    loc = {}
    exec(f'import {self.name} as suite', {}, loc)
    suite = loc['suite']
    case_keys = list(sorted(filter(lambda x: x.startswith('benchmark_'), dir(suite))))
    self.cases = {k: getattr(suite, k) for k in case_keys}
    
  def print(self):
    print(f'Suite {self.name}:')
    for b in self.cases:
      print(' *', b)

class TaichiBenchmark:
  def __init__(self):
    self.suites = []
    for f in sorted(os.listdir('.')):
      if f != 'run.py' and f.endswith('.py') and f[0] != '_':
        self.suites.append(Suite(f))
        
        
  def pprint(self):
    for s in self.suites:
      s.print()


b = TaichiBenchmark()
b.pprint()

def run():
  return