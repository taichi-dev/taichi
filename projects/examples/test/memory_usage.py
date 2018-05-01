import taichi as tc

# ti plot mem.txt   should show the result

tc.start_memory_monitoring('mem.txt')
a = []
while True:
  a.append(123)
