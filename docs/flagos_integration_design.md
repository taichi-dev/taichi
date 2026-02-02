# Taichi FlagOS 后端集成设计文档

## 概述

本文档描述了将 FlagOS 集成到 Taichi 编译器的设计方案。FlagOS 是面向多种 AI 芯片的统一开源系统软件栈，通过集成 FlagOS，Taichi 可以实现对多种国产 AI 芯片的支持。

## 架构设计

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Taichi Frontend                        │
│                    (Python API / DSL)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Taichi Compiler                          │
│         (AST Transformation / SNode / Optimization)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Code Generation                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────────┐   │
│  │  LLVM   │ │  CUDA   │ │  AMDGPU │ │     FlagOS       │   │
│  │ (x64)   │ │ (NVIDIA)│ │  (AMD)  │ │ (Multi-Chip)     │   │
│  └─────────┘ └─────────┘ └─────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FlagOS Backend                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   FlagTree Compiler                      ││
│  │    (Unified MLIR/LLVM-based Compiler for AI Chips)       ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │  MLU    │ │  Ascend │ │  DCU    │ │  GCU    │ │  ...   │ │
│  │(Cambricon)│(Huawei) │ │(Hygon)  │ │(Enflame)│ │        │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2. 集成层次

FlagOS 后端作为 LLVM 后端的一种变体实现：

1. **RHI 层** (`taichi/rhi/flagos/`): 硬件抽象层
   - 实现 `FlagosDevice` 继承 `LlvmDevice`
   - 通过 FlagOS 运行时 API 进行内存管理和内核启动

2. **代码生成层** (`taichi/codegen/flagos/`): LLVM IR 生成
   - 实现 `TaskCodeGenFlagOS` 继承 `TaskCodeGenLLVM`
   - 针对 FlagOS/FlagTree 的特殊优化和代码生成

3. **运行时层** (`taichi/runtime/program_impls/flagos/`): 程序执行
   - 实现 `FlagosProgramImpl` 继承 `LlvmProgramImpl`
   - 集成 FlagOS 运行时和 JIT 编译

## 实现细节

### 3.1 架构定义

添加 `flagos` 到架构枚举：

```cpp
// taichi/inc/archs.inc.h
PER_ARCH(flagos)  // FlagOS: Unified AI Chip Backend
```

更新架构相关函数：

```cpp
// taichi/rhi/arch.cpp
bool arch_uses_llvm(Arch arch) {
  return (arch == Arch::x64 || arch == Arch::arm64 || arch == Arch::cuda ||
          arch == Arch::dx12 || arch == Arch::amdgpu || arch == Arch::flagos);
}

bool arch_is_gpu(Arch arch) {
  return !arch_is_cpu(arch) || arch == Arch::flagos;
}
```

### 3.2 RHI 设备实现

`FlagosDevice` 核心功能：

```cpp
class FlagosDevice : public LlvmDevice {
 public:
  // 内存管理
  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  void dealloc_memory(DeviceAllocation handle) override;

  // FlagOS 运行时接口
  void *get_memory_addr(DeviceAllocation devalloc) override;
  std::size_t get_total_memory() override;

  // 内核执行
  Stream *get_compute_stream() override;
  void wait_idle() override;

 private:
  // FlagOS 上下文和驱动
  std::shared_ptr<FlagosContext> context_;
};
```

### 3.3 代码生成

`TaskCodeGenFlagOS` 关键特性：

1. **目标三元组**: 通过 FlagTree 指定目标芯片
2. **内建函数**: 使用 FlagOS 提供的数学库
3. **并行原语**: 适配 FlagOS 的线程模型

```cpp
class TaskCodeGenFlagOS : public TaskCodeGenLLVM {
  void visit(OffloadedStmt *stmt) override;
  void create_offload_range_for(OffloadedStmt *stmt) override;

  // SPMD 信息获取
  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override;
};
```

### 3.4 运行时集成

FlagOS 运行时通过 FlagTree 的 JIT 接口：

```cpp
class FlagosProgramImpl : public LlvmProgramImpl {
 public:
  void materialize_runtime(KernelProfilerBase *profiler,
                          uint64 **result_buffer_ptr) override;

  // FlagOS 特定初始化
  void initialize_flagos_backend();

 private:
  std::unique_ptr<FlagosRuntime> flagos_runtime_;
};
```

## FlagOS API 集成

### 4.1 FlagTree 编译器接口

```cpp
// FlagTree API 封装
namespace flagos {

class FlagTreeCompiler {
 public:
  // 编译 LLVM IR 到目标芯片代码
  std::vector<uint8_t> compile(const std::string &llvm_ir,
                               const std::string &target_chip);

  // 获取支持的芯片列表
  std::vector<std::string> get_supported_chips();
};

class FlagosRuntime {
 public:
  // 内存分配
  void* allocate_memory(size_t size, bool managed);
  void free_memory(void* ptr);

  // 内核启动
  void launch_kernel(const std::string &kernel_name,
                    void **args,
                    uint32_t grid_dim,
                    uint32_t block_dim);

  // 同步
  void synchronize();
};

} // namespace flagos
```

### 4.2 芯片配置

支持通过环境变量或配置指定目标芯片：

```python
import taichi as ti

# 方式1: 通过 arch 指定
ti.init(arch=ti.flagos)

# 方式2: 通过环境变量
# export TI_FLAGOS_CHIP=mlu370
# export TI_FLAGOS_CHIP=ascend910

ti.init(arch=ti.flagos,
        flagos_chip="mlu370")  # 指定目标芯片
```

## 构建配置

### 5.1 CMake 配置

```cmake
# CMakeLists.txt
option(TI_WITH_FLAGOS "Build with FlagOS support" OFF)

if(TI_WITH_FLAGOS)
  find_package(FlagOS REQUIRED)
  add_definitions(-DTI_WITH_FLAGOS)

  # FlagOS RHI
  add_subdirectory(taichi/rhi/flagos)

  # FlagOS Codegen
  add_subdirectory(taichi/codegen/flagos)

  # FlagOS Runtime
  add_subdirectory(taichi/runtime/program_impls/flagos)
endif()
```

### 5.2 依赖要求

- FlagOS SDK >= 1.5
- FlagTree >= 0.8
- LLVM >= 15.0
- C++17 编译器

## 开发路线图

### 阶段 1: 基础架构 (MVP)
- [x] 添加 flagos 架构定义
- [x] 实现基础 FlagosDevice
- [x] 集成 FlagTree LLVM 编译器
- [x] 支持基础内存操作

### 阶段 2: 内核执行
- [ ] 实现内核编译和加载
- [ ] 支持 range_for 内核
- [ ] 支持 struct_for 内核
- [ ] 实现原子操作

### 阶段 3: 高级特性
- [ ] 支持 SNode 稀疏数据结构
- [ ] 实现设备间内存拷贝
- [ ] 支持 Profiler
- [ ] AOT (Ahead-of-Time) 编译

### 阶段 4: 多芯片支持
- [ ] 寒武纪 MLU 系列
- [ ] 华为昇腾 Ascend 系列
- [ ] 海光 DCU 系列
- [ ] 燧原 GCU 系列

## 测试策略

### 6.1 单元测试
- RHI 设备功能测试
- 内存分配/释放测试
- 内核启动测试

### 6.2 集成测试
- 运行 Taichi 官方示例
- 性能基准测试
- 多芯片兼容性测试

### 6.3 CI/CD
```yaml
# 示例 CI 配置
flagos_tests:
  runs-on: flagos-ci-runner
  steps:
    - name: Test MLU Backend
      env:
        TI_FLAGOS_CHIP: mlu370
      run: python -m pytest tests/test_flagos_mlu.py

    - name: Test Ascend Backend
      env:
        TI_FLAGOS_CHIP: ascend910
      run: python -m pytest tests/test_flagos_ascend.py
```

## 相关链接

- [FlagOS GitHub](https://github.com/flagos-ai)
- [FlagTree 编译器](https://github.com/flagos-ai/flagtree)
- [Taichi 后端开发文档](https://docs.taichi-lang.org/docs/master/hackers_guide)
