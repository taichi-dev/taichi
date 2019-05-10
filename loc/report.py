import os

for f in os.listdir('.'):
    if f.endswith('.cpp'):
        os.system('clang-format-6.0 -i {}'.format(f))

os.system('wc *.cpp')