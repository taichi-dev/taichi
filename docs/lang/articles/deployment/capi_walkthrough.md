# C-API Walkthrough

C-API 提供了最核心的 Taichi Runtime 功能：创建后端设备、分配内存、执行 Kernel 以及前后端互传数据的基础设施。这里给出一个 Fractal 的例子，让读者熟悉一下 C-API 的工作流，看看 Taichi AOT Module 在 C++ 侧的部署代码大概会是什么样子。

## Python: 编译 AOT Module

首先我们运行脚本，编译 Fractal 的 compute graph。

```python
import taichi as ti

ti.init(arch=ti.vulkan)

@ti.kernel
def fractal(t: ti.f32, canvas: ti.types.ndarray(field_dim=2)):
    # ...

# 创建 Argment Symbol，每个 Symbol 都代表了一个传入 kernel 的参数。
# Compute graph 需要 Symbol 来辨认被传入不同 Kernel 的同一个对象。
sym_t = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "t", ti.f32, element_shape=())
sym_canvas = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "canvas", ti.f32, field_dim=2, element_shape=())

# 创建一个计算图，然后将 argument symbol 代入 kernel 进行预编译。
gb = ti.graph.GraphBuilder()
gb.dispatch(fractal, sym_t, sym_canvas)
graph = gb.compile()

# 创建 AOT Module，插入 Compute Graph 并取名为 “fractal” 。打包保存到 module 目录。
mod = ti.aot.Module(ti.vulkan)
mod.add_graph('fractal', graph)
mod.save("module", "")
```

完整的 Python 脚本可以在[这里](https://github.com/PENGUINLIONG/Minimalist-TaichiAOT/blob/fractal-cpp/app.py)找到。

## C++: 加载 AOT Module 并调度执行
首先我们引入 taichi 的部署头文件。

```cpp
#include <taichi/cpp/taichi.hpp>
```

> ⚠️ 注意
> 不要直接引入 `taichi/taichi_core.h` 等头文件。先引入 `taichi/taichi.h` 这个 master header，然后用 `TI_WITH_VULKAN` 等宏控制是否引入 non-core 功能。

创建一个对应 Arch 的 Runtime。上面编译的目标 Arch 是 Vulkan，这里我们也一样创建 Vulkan 的 Runtime。然后从 `module` 目录加载 AOT Module，取出计算图 `fractal`。

```cpp
ti::Runtime runtime(TI_ARCH_VULKAN);
ti::AotModule aot_module = runtime.load_aot_module("module");
ti::ComputeGraph fractal = aot_module.get_compute_graph("fractal");
```

为画布分配内存，构造一个 ND-array，然后绑定参数。注意 fractal demo 的分形会随时间扭动，我们的部分参数（如 t）可能会每帧变化，但 canvas 这种每帧不变的参数可以提前绑定上去。

```cpp
ti::NdArray<float> canvas = runtime.allocate_ndarray<float>({ WIDTH, HEIGHT }, {}, true);

fractal["canvas"] = canvas;
```

在 render loop 里绑定每帧变化的参数，launch 计算图，等待设备执行返回，然后绘制上屏。

```cpp
for (uint32_t i = 0; i < 10000; ++i) {
  fractal["t"] = 0.03f * i;

  fractal.launch();
  runtime.wait();

  draw(runtime, canvas);
  std::this_thread::sleep_for(0.03s);
}
```

具体的代码可以在[这里](https://github.com/PENGUINLIONG/Minimalist-TaichiAOT/blob/fractal-cpp/app.cpp)找到。

接下来只需要编译并链接到 libtaichi_c_api.so 就可以运行了。

## Tl;dr

简单来说，部署一个 AOT 程序一般需要如下几个步骤：

1. 创建 Runtime
2. 加载 AOT module 并提取 Compute Graph
3. 分配内存（Memory, Image）并流送输入数据
4. 构建参数表
5. Launch & Wait