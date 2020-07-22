import taichi as ti

with ti.ActionRecord('record.yml'):
    ti.init(ti.cc, log_level=ti.DEBUG)

    @ti.kernel
    def func():
        print(233)


    func()
