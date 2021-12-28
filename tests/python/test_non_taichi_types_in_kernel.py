import taichi as ti


@ti.test()
def test_subscript_user_classes_in_kernel():
    class MyList:
        def __init__(self, elements):
            self.elements = elements

        def __getitem__(self, index):
            return self.elements[index]

    @ti.kernel
    def func():
        for i in ti.static(range(3)):
            print(a[i])

    a = MyList([1, 2, 3])
    func()
