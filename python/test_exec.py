#def f(x):
#  return x + 1

def main():
  exec('def f(x): return x', globals(), locals())
  print(globals())
  print(locals())
  print(locals()['f'](1))

if __name__ == '__main__':
  main()
