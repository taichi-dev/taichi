# This file generates a taichi header (analgamation)

import os
import sys
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
  
include_template = r'#\s?include.*([<"])(.*)[>"]'
search_directories = ['include', 'external/include']

class Amalgamator:
  def __init__(self):
    self.files = expand_files(files_to_include)
    self.included = set()
    self.output_f = open('build/taichi.h', 'w')
    print("#define TC_INCLUDED", file=self.output_f)
    print("#define TC_AMALGAMATED", file=self.output_f)
    
  def include(self, fn):
    if fn in self.included:
      print('Skipping {}'.format(fn))
      return
    self.included.add(fn)
    print('Including {}'.format(fn))
    with open(fn, 'r') as f:
      lines = f.readlines()
      for l in lines:
        l = l.rstrip()
        if l == '#pragma once':
          continue
        match = re.search(include_template, l)
        need_expand = False
        if match:
          assert match.group(1) in ['\"', '<']
          local = (match.group(1) == '\"')
          includee = match.group(2)
          
          local_dir = fn[:fn.rfind('/')]
          local_search_directories = search_directories
          if local:
            local_search_directories = [local_dir] + local_search_directories
          # Search for file
          found = False
          for d in local_search_directories:
            suspect_includee = os.path.join(d, includee)
            if os.path.exists(suspect_includee):
              found = True
              need_expand = True
              includee = suspect_includee
            else:
              pass # Should be system header
          if not found:
            print("  X - Classified as stdc++ header: {}".format(includee))
            #print("  ({})".format(l))
            need_expand = False
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