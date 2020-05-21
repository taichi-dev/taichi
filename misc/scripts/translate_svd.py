import taichi as tc
import os
import re

with open(tc.get_repo_directory() + '/include/taichi/math/sifakis_svd.h') as f:
    lines = f.readlines()
    lines = list(map(lambda x: x.strip(), lines))

lines = list(filter(bool, lines))

variables = []

l_var = lines.index('// var')
l_compute = lines.index('// compute')
l_end = lines.index('// end')
l_output = lines.index('// output')

print(l_var, l_compute, l_end)

variables = []

for l in range(l_var, l_compute - 1, 4):
    var = lines[l + 4][2:-1]
    print(var)
    variables.append(var)

print('variables:', variables)

f = open(
    tc.get_repo_directory() + '/projects/taichi_lang/src/tests/svd_body.h',
    'w')

print(
    'template <typename Tf=float32, typename Ti=int32> std::tuple<Matrix, Matrix, Matrix> sifakis_svd(const Matrix &a) {',
    file=f)

print()
print()
print('code...')
print()
print()

print('''
static_assert(sizeof(Tf) == sizeof(Ti), "");
constexpr Tf Four_Gamma_Squared = 5.82842712474619f;
constexpr Tf Sine_Pi_Over_Eight = 0.3826834323650897f;
constexpr Tf Cosine_Pi_Over_Eight = 0.9238795325112867f;
''',
      file=f)

for var in variables:
    print("auto {} = Var(Tf(0.0));".format(var), file=f)


def to_var(s):
    if len(s) == 3 and s[0] == 'a':
        return 'a({}, {})'.format(int(s[1]) - 1, int(s[2]) - 1)
    if s[:5] == 'rsqrt':
        return s[:-3] + ')'
    if s.find('-') != -1:
        return s
    if len(s) > 3 and s[-3] == '.':
        return s
    assert s[-3:] != '.ui', s
    if s[:-2] in variables:
        s = s[:-2]
        return s
    else:
        return 'Expr(' + s + ')'


def to_var_ui(s):
    assert s[-3:] == '.ui', s
    s = s[:-3]
    if s in variables:
        return s
    else:
        if s[0] == '~':
            s = '~bit_cast<Ti>({})'.format(s[1:])
        return 'Expr(' + s + ')'


for l in lines[l_compute + 1:l_end]:
    if l[-1] == ';':
        l = l[:-1]
    tokens = l.split()
    if len(tokens) == 3 and tokens[1] == '=':
        if tokens[0][-2:] == '.f':
            if re.search('[a-zA-Z]', tokens[2]) and ('.' not in tokens[2]
                                                     or 'sqrt' in tokens[2]
                                                     or 'S' in tokens[2]):
                print("{} = {};".format(to_var(tokens[0]), to_var(tokens[2])),
                      file=f)
            else:
                print("{} = Tf({});".format(to_var(tokens[0]),
                                            to_var(tokens[2])),
                      file=f)
        continue
    if len(tokens) == 5 and tokens[1] == '=' and len(tokens[3]) == 1:
        # a = b + c ...
        if tokens[0][-2:] == '.f':
            print("{} = {} {} {};".format(to_var(tokens[0]), to_var(tokens[2]),
                                          tokens[3], to_var(tokens[4])),
                  file=f)
        else:
            op = tokens[3]
            if op == '^':
                op_name = 'xor'
            elif op == '|':
                op_name = 'or'
            elif op == '&':
                op_name = 'and'
            else:
                assert False
            print("{} = svd_bitwise_{}<Tf, Ti>({}, {});".format(
                to_var_ui(tokens[0]), op_name, to_var_ui(tokens[2]),
                to_var_ui(tokens[4])),
                  file=f)
        continue
    if len(tokens) == 9 and tokens[1] == '=' and tokens[5] == '?' and tokens[
            7] == ':':
        # a = b ? a : d
        print(
            "{} = bit_cast<Tf>(select({} {} {}, Expr(Ti(int32({}))), Expr(Ti({}))));"
            .format(to_var_ui(tokens[0]), tokens[2][1:-2], tokens[3],
                    tokens[4][:-3], tokens[6], tokens[8]),
            file=f)
        continue
    if len(tokens) == 4 and tokens[1] == '=' and tokens[2][:8] == 'std::max':
        # a = std::max(...)
        # continue
        print("{} = max({}, {});".format(to_var(tokens[0]), tokens[2][9:-3],
                                         tokens[3][:-3]),
              file=f)
        continue
    if tokens[0] == 'for':
        print('For (0, 5, [&] (Expr sweep) {', file=f)
        continue
    if tokens[0] == '}':
        print('});', file=f)
        continue
    print(tokens)
    print(l)

print("Matrix u(3, 3), v(3, 3), sigma(3);", file=f)

for l in lines[l_end + 1:l_output - 3]:
    print("{}({}, {}) = {};".format(l[0],
                                    int(l[1]) - 1,
                                    int(l[2]) - 1, l[-7:-3]),
          file=f)

for l in lines[l_output - 3:l_output]:
    print("sigma({}) = {};".format(int(l[5]) - 1, l[-7:-3]), file=f)

print('return std::make_tuple(u, sigma, v);', file=f)
print("}", file=f)

f.close()

os.system('clang-format-6.0 -i {}'.format(
    tc.get_repo_directory() + '/projects/taichi_lang/src/tests/svd_body.h'))
