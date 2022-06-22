---
sidebar_position: 0
---

# Why a new programming language

Imagine you want to write a new particle-based fluid algorithm. You start simple: It does not take much time before you find some C++/CUDA reference work online (or derive a reference from your labmate, <s>un</s>fortunately). `cmake .. && make`, you type. Oops, CMake throws out an error due to a random incompatible third-party library. Install and rebuild it ... Problem solved? You run the program, which immediately segfaults (without any stack trace, of course). Then you start gazing at the code, place the necessary asset files at the right place, fix a few dangling pointers, and finally, rerun the program. It works ... if you does not plug in your revised algorithm. Now you face another big fight with the GPU or CPU code. At the end of the day, you are just buried in the language details.

If all this sounds too familiar to you, congratulations! You are likely to find the antidote here.

Born from the MIT CSAIL lab, Taichi is a domain-specific language (DSL) designed to make life easier for computer graphics researchers, helping them quickly implement visual computing and physics simulation algorithms that are executable on GPU. Taichi is innovative in that it is embedded in Python and uses modern just-in-time (JIT) frameworks (for example, LLVM and SPIR-V) to offload the Python source code to native GPU or CPU instructions, offering performance at both development time and runtime.

## How Taich differs from other DSLs

Frankly, a DSL with a Python frontend is not something new. Frameworks like Halide, PyTorch, and TVM have matured into the de facto standards in such areas as image processing and deep learning (DL). What distinguishes Taichi the most from these frameworks is its imperative programming paradigm. As a DSL, Taichi is not particularly specialized in a certain computing pattern, and this provides better flexibility. While one may argue that flexibility usually comes at the cost of inadequante optimization, we often find this is not the case with Taichi for the following reasons:

* Taichi's workload typically does *not* exhibit an exploitable pattern (such as element-wise operations). This means that the arithmetic intensity is bounded anyway. By simply switching to the GPU backend, one can already enjoy a nice performance gain.
* Unlike traditional DL frameworks, where operators are simple math expressions and have to be fused at the graph level to achieve higher arithmetic intensity, Taichi, with the imperative paradigm, makes it quite easy to carry out a large amount of computation in a single kernel. We call such a kernel a *mega-kernel*.
* Taichi heavily optimizes the source code using various compiler technologies, including common subexpression elimination, dead code elimination, and control flow graph analysis. The optimization is backend-neutral because Taichi hosts its own intermediate representation (IR) layer.
* JIT compilation provides additional optimization opportunities.

## Till now, not even the whole picture

Taichi goes well beyond a Python JIT transpiler. One of the initial design goals is to *decouple the computation from the data structures*. To achieve this goal, Taichi provides a mechanism called *Snode* (/ˈsnoʊd/), which is essentially a set of generic data containers. SNodes can be used to compose hierarchical, dense or sparse, multi-dimensional fields conveniently. Switching between array-of-structures and structure-of-arrays layouts is usually a matter of ≤10 lines of code. This has inspired many use cases in numerical simulation. If you are interested to learn more about the mechanism, please check out [Fields (advanced)](../advanced/layout.md), [Sparse spatial data structures](../advanced/sparse.md), or [the original Taichi paper](https://yuanming.taichi.graphics/publication/2019-taichi/taichi-lang.pdf).

The concept of decoupling is further extended to the type system. Nowadays, as GPU memory capacity and bandwidth increasingly pose major bottlenecks, it is vital to be able to pack more data per memory unit. Taichi introduced customizable quantized types in 2021, allowing user-defined fixed-point or floating-point numbers with arbitrary bits (still needs to be under 64). This makes possible an MPM simulation of over 400 million particles on a single GPU device. Learn more details in [the QuanTaichi paper](https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf).

## A user-/developer-friendly language

Taichi is intuitive. If you know Python, you know Taichi. If you write Taichi, you awaken your GPU (or CPU as a fallback). Ever since its debut, this simple idea has gained so much popularity that contributors keep experimenting on and incorporating new backends, including Vulkan, OpenGL, and DirectX (working in progress). Without our strong and dedicated community, Taichi would never have been where it is now.

We are never afaid to aim high. Going forward, we see pleanty more opportunities where Taichi can make a difference. We would like to share some (definitely not exhaustive) application scenarios with you and hopefully might give a taste of our vision.

## Where Taichi can offer relief

### Academia

The nature of research makes it a painful fact that 90% of the research code is trashed because assumptions keep being broken and ideas keep being iterated. Swift coding without considering performance may lead to incorrect conclusions, and pre-matured code optimization can be a waste of time and often produces a tangled mess. Therefore, a language offering high performance and productivity can tremendously benefit research projects.

Taichi keeps embracing the academia. The key features we have (or plan to have) for high-performance computing research projects include small-scale linear algebra (inside kernels), large-scale sparse systems, and efficient neighbor accessing for both structured and unstructured data. Taichi also provides an automatic differentiation module via source code transformation (at IR level), making it a sweet differentiable simulation tool for machine learning projects.

### Apps & game engine integration

One huge advantange of Taichi lies in its portability, thanks to the support for a wide variety of backends. During the development process, we have recognized the increasing demand from our industry users for multi-platform packaging and deployment.

The experimental demo Below shows a seamless integration of Taichi and Unity. By exporting Taichi kernels as SPIR-V shaders, we can easily import them into a Unity project.

![](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/unity_fluid.gif)

### General-purpose computing

While originally designed for physics simulation, Taichi has found its application in many other areas that can be boosted by GPU general-purpose computing.

* [taichimd](https://github.com/victoriacity/taichimd): Interactive, GPU-accelerated Molecular (& Macroscopic) Dynamics using the Taichi programming language.
* [TaichiSLAM](https://github.com/xuhao1/TaichiSLAM): a 3D Dense mapping backend library of SLAM based on Taichi-Lang, designed for the aerial swarm.
* [Stannum](https://github.com/ifsheldon/stannum): Fusing Taichi into PyTorch.

### Maybe a new frontend?

The benefit of adopting the compiler approach is that you can decouple the frontend from the backend. Taichi is *currently* embedded in Python, but who says it needs to stay that way? Stay tuned [:](https://taichi-js.com/playground)-)
