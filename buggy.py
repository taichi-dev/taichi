import taichi as ti

@ti.kernel
def buggy():
        ret = 0  # 0 is a integer, so ret is typed as i32
        for i in range(4):
                ret += 0.1 * i  # i32 += f32, the result is still stored in i32!
                print(ret)  # will shows 0

buggy()
