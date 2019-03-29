import taichi as tc
import os


with open(tc.get_repo_directory() + '/include/taichi/math/sifakis_svd.h') as f:
    lines = f.readlines()
    lines = list(map(lambda x: x.strip(), lines))

lines = list(filter(bool, lines))

variables = []

l_var = lines.index('// var')
l_compute = lines.index('// compute')
l_end = lines.index('// end')

print(l_var, l_compute, l_end)

variables = []

for l in range(l_var, l_compute - 1, 4):
    var = lines[l + 4][2:-1]
    print(var)
    variables.append(var)

print('variables:', variables)

print()
print()
print('code...')
print()
print()

for l in lines[l_compute + 1:l_end]:
    tokens = l.split()
    if len(tokens) == 3 and tokens[1] == '=':
        # a = xxxx
        continue
    if len(tokens) == 5 and tokens[1] == '=' and len(tokens[3]) == 1:
        # a = b + c ...
        continue
    if len(tokens) == 9 and tokens[1] == '=' and tokens[5] == '?' and tokens[7] == ':':
        # a = b ? a : d
        continue
    if len(tokens) == 4 and tokens[1] == '=' and tokens[2][:8] == 'std::max':
        # a = std::max(...)
        continue
    if tokens[0] == 'for':
        continue
    if tokens[0] == '}':
        continue
    print(tokens)
    print(l)
