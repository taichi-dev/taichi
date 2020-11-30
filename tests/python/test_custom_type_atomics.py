import taichi as ti

# @ti.test(require=ti.extension.quant, debug=True, cfg_optimization=False)
# def test_custom_int_load_and_store():

def main():
    ti.init(arch=ti.cpu, debug=True, cfg_optimization=False, print_ir=True)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu2 = ti.type_factory_.get_custom_int_type(2, False)
    
    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu2)
    
    ti.root._bit_struct(num_bits=32).place(x, y)
    
    x[None] = 3
    
    # TODO: unsigned
    # y[None] = 1
    
    @ti.kernel
    def foo():
        for i in range(10):
            x[None] += 4
    
    foo()
    
    assert x[None] == 43

main()
