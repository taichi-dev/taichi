import taichi as ti

@ti.test(require=ti.extension.quant)
def test_shared_exponents():
    
    a = ti.field
