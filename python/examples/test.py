import taichi as tc

tc.core.test()

try:
    tc.core.test_raise_error()
except Exception as e:
    print 'Exception:', e

print 'Testing finished.'
