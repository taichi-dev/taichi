import time
import taichi as ti
from utils import *
from baseline_fill import fill
from baseline_reduction import reduction

ti.init(kernel_profiler=True)

test_cases = [fill,reduction]
test_archs = [ti.cuda]#, ti.cpu]
test_dtype = [ti.i32, ti.i64, ti.f32, ti.f64]
test_dsize = [
    4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
    268435456, 1073741824
]
# [  4KB,  16KB,  64KB,  256KB,     1MB,     4MB,     16MB,     64MB,     256MB,     1024MB ]



class testResult:
    test_arch = None
    data_type = None
    data_size = None
    min_time_in_us = []
    def __init__(self, arch, dtype, dsize):
        self.test_arch = arch
        self.data_type = dtype
        self.data_size = dsize

class caseImpl:
    func = None
    name = None
    env = None
    device = None
    archs = None
    data_type  = None
    data_size  = None
    test_result = []
    def __init__(self, func, archs, data_type, data_size):
        self.func = func
        self.name = func.__name__
        self.archs = archs
        self.data_type = data_type
        self.data_size = data_size
    def run(self):
        for arch in self.archs:
            ti.reset() 
            ti.init(kernel_profiler=True, arch=arch)
            for dtype in self.data_type:
                print("%s.%s.%s" %(self.func.__name__, ti.core.arch_name(arch), dtype2str[dtype]))
                self.test_result.append(testResult(arch,dtype,self.data_size))
                for size in self.data_size:
                    print( "TEST data size = %s #####################" % (size2str(size)))
                    self.test_result[-1].min_time_in_us.append(self.func(arch, dtype, size, 10))
                    time.sleep(1)
    def print(self):
        i = 0
        for arch in self.archs:
            for dtype in self.data_type:
                for idx in range(len(self.data_size)):
                        print("kernel:[%s] arch:[%s] type:[%s] size:[%s] >>> time:[%4.4f]" 
                        %(self.func.__name__, ti.core.arch_name(arch), dtype2str[dtype], size2str(self.data_size[idx]), self.test_result[i].min_time_in_us[idx]))
                i=i+1


def performance():
    for case in testCaseImpl:
        case.run()

def performance_print():
    for case in testCaseImpl:
        case.print()

testCaseImpl = []
for case in test_cases:
    testCaseImpl.append(caseImpl(case,test_archs,test_dtype,test_dsize))

def baseline():
    performance()
    performance_print()