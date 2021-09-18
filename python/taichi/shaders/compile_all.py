import os
import sys
from os import listdir
from os.path import isfile, join

# use this script to compile all shaders before running `pip3 install .`

if __name__ == "__main__":
    files = [f for f in listdir('.') if isfile(join('.', f))]
    for f in files:
        suffix = f.split('.')[-1]
        if suffix in ['frag', 'vert', 'geom']:
            name = f.split('.')[0]
            output = f'{name}_{suffix}.spv'
            command = f'glslangValidator -V -o {output} {f}'
            print(f"running: {command}")
            os.system(command)
