import astpretty
import ast

expr = """
@taichi
def foo(x):
    global s
    
    local.r = 1
    
    x = float32(1)
    x = float64(1)
    x = int(1)
    y = x
    x[i] = (1 + 2 - 3)
    
    if constexpr(dim == 2):
        print(2)
    else:
        print(1)
        
    unroll()
    for i, j in range(x):
        print("hello world")
        SLP(4)
"""
p = ast.parse(expr)

# p.body[0].body = [ast.parse("return 42").body[0]] # Replace function body with "return 42"

astpretty.pprint(p)