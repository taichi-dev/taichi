import taichi as tc

def main():
    tc.info('test_logging, a = {}, b = {b}', 10, b=123)
    print(0)

tc.redirect_print_to_log()
main()
print(123)
