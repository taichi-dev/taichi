# This file generates a taichi header (analgamation)

import os
import re

files_to_include = [
  'include/taichi/common/*',
  'src/system/timer.cpp'
]

def expand_files(files):
  new_files = []
  for f in files:
    if f[-1] == '*':
      for header in os.listdir(f[:-1]):
        new_files.append(f[:-1] + header)
    else:
      assert f.endswith('.h') or f.endswith('.cpp') or f.endswith('.cc') or f.endswith('.c')
      new_files.append(f)
  return new_files
  
include_template = r'#include.*([<"](.*)[>"])'

class Amalgamator:
  def __init__(self):
    self.files = expand_files(files_to_include)
    self.included = set()
    self.output_f = open('build/taichi.h', 'w')
    
  def include(self, fn):
    if fn in self.included:
      print('Skipping {}'.format(fn))
      return
    self.included.add(fn)
    print('Including {}'.format(fn))
    with open(fn, 'r') as f:
      lines = f.readlines()
      for l in lines:
        l = l.strip()
        if l == '#pragma once':
          continue
        match = re.search(include_template, l)
        need_expand = False
        if match:
          local = (match.group(1) == '\"')
          includee = match.group(2)
          if local:
            need_expand = True
            includee = fn[:fn.rfind('/')] + includee
          else:
            # taichi headers or not?
            if includee.startswith('taichi'):
              need_expand = True
              includee = 'include/' + includee
            else:
              pass # Should be system header
        if need_expand:
          self.include(includee)
        else:
          print(l, file=self.output_f)
  
  def run(self):
    for f in self.files:
      self.include(f)

def main():
  ama = Amalgamator()
  ama.run()

if __name__ == '__main__':
  main()
  os.system('g++ build/taichi.h -o build/taichi -std=c++14')