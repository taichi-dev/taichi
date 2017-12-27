import taichi as tc


def main():
  tc.info('test_logging, a = {}, b = {b}', 10, b=123)
  print(0)


tc.duplicate_stdout_to_file('a.log')

tc.redirect_print_to_log()
main()
print(123)
