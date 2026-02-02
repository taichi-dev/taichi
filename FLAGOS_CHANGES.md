# FlagOS Backend Integration for Taichi - Change Log

## Overview
This change adds support for FlagOS (unified AI chip software stack) to Taichi, enabling Taichi programs to run on various AI chips including MLU (Cambricon), Ascend (Huawei), DCU (Hygon), and GCU (Enflame).

## Changes Summary

### 1. Core Architecture Changes

#### Modified Files:
- `taichi/inc/archs.inc.h` - Added `flagos` architecture definition
- `taichi/rhi/arch.cpp` - Updated architecture functions to include flagos
- `taichi/program/compile_config.h` - Added `flagos_chip` configuration option
- `taichi/python/export_lang.cpp` - Exposed flagos configuration to Python API
- `taichi/program/program.cpp` - Added FlagosProgramImpl instantiation logic

### 2. Build System Changes

#### Modified Files:
- `cmake/TaichiCore.cmake` - Added `TI_WITH_FLAGOS` option and build configuration
- `taichi/rhi/CMakeLists.txt` - Added FlagOS RHI subdirectory

### 3. New Files - FlagOS RHI Device Layer

**Location:** `taichi/rhi/flagos/`

| File | Description |
|------|-------------|
| `flagos_device.h` | FlagOS device class definition inheriting from LlvmDevice |
| `flagos_device.cpp` | Device implementation: memory management, data transfer, kernel launch |
| `CMakeLists.txt` | Build configuration for FlagOS RHI |

**Key Features:**
- Multi-chip support (MLU370, MLU590, Ascend910, Ascend310, DCU, GCU, generic)
- Environment variable `TI_FLAGOS_CHIP` for chip selection
- Integration with LLVM device memory pool

### 4. New Files - FlagOS Code Generation Layer

**Location:** `taichi/codegen/flagos/`

| File | Description |
|------|-------------|
| `codegen_flagos.h` | Code generator header file |
| `codegen_flagos.cpp` | LLVM IR generation implementation for AI chips |
| `CMakeLists.txt` | Build configuration |

**Key Features:**
- Inherits from TaskCodeGenLLVM
- SPMD (Single Program Multiple Data) execution model support
- FlagOS-specific reduction operations
- Optimized grid/block dimension configuration

### 5. New Files - FlagOS Program Implementation Layer

**Location:** `taichi/runtime/program_impls/flagos/`

| File | Description |
|------|-------------|
| `flagos_program.h` | Program implementation header |
| `flagos_program.cpp` | Runtime integration with FlagOS |
| `flagos_kernel_compiler.h` | Kernel compiler header |
| `flagos_kernel_compiler.cpp` | Kernel compiler implementation |
| `flagos_kernel_launcher.h` | Kernel launcher header |
| `flagos_kernel_launcher.cpp` | Kernel launcher implementation |
| `CMakeLists.txt` | Build configuration |

**Key Features:**
- Extends LlvmProgramImpl
- FlagOS-specific kernel compilation and launching
- Integration with FlagTree compiler (placeholder for SDK integration)

### 6. New Files - Examples and Documentation

**Location:** `examples/flagos/`

| File | Description |
|------|-------------|
| `fractal_flagos.py` | Julia set fractal computation example |
| `matmul_flagos.py` | Matrix multiplication benchmark |
| `README.md` | Usage documentation for FlagOS backend |

**Location:** `docs/`

| File | Description |
|------|-------------|
| `flagos_integration_design.md` | Detailed design documentation |

## Build Instructions

### CMake Configuration
```bash
mkdir build && cd build
cmake .. -DTI_WITH_FLAGOS=ON -DTI_WITH_LLVM=ON
make -j$(nproc)
```

### Environment Setup
```bash
export TI_FLAGOS_CHIP=mlu370  # or ascend910, dcu, gcu, generic
```

## Usage Example

```python
import taichi as ti

# Initialize FlagOS backend
ti.init(arch=ti.flagos, flagos_chip="mlu370")

@ti.kernel
def compute():
    for i in range(1000000):
        # Parallel computation
        pass

compute()
```

## Architecture Integration Flow

```
Taichi DSL (Python)
       ↓
Taichi IR
       ↓
LLVM IR (TaskCodeGenFlagOS)
       ↓
FlagTree Compiler (via FlagOS SDK)
       ↓
Target AI Chip Binary (MLU/Ascend/DCU/GCU)
```

## Testing

### Supported Chips (Planned)
- MLU370, MLU590 (Cambricon)
- Ascend910, Ascend310 (Huawei)
- DCU (Hygon)
- GCU (Enflame)
- generic (fallback for testing)

### Test Commands
```bash
# Run fractal example
python examples/flagos/fractal_flagos.py

# Run matrix multiplication benchmark
python examples/flagos/matmul_flagos.py
```

## Dependencies

### Required for Building
- LLVM >= 15.0
- C++17 compiler
- FlagOS SDK >= 1.5 (for full functionality)
- FlagTree >= 0.8 (for full functionality)

### Optional Dependencies
- FlagGems (for optimized operators)
- FlagCX (for multi-chip communication)

## Known Limitations

1. **SDK Integration**: Current implementation uses stub interfaces for FlagOS SDK. Full functionality requires FlagOS SDK integration.

2. **Kernel Compilation**: LLVM IR to chip binary compilation is placeholder and needs FlagTree compiler integration.

3. **Math Libraries**: Device-side math functions use LLVM defaults. FlagOS-optimized math library integration pending.

## Future Work

### Phase 1: SDK Integration
- [ ] Integrate FlagTree C++ API
- [ ] Implement chip-specific code generation
- [ ] Add FlagOS runtime memory management

### Phase 2: Chip-Specific Optimizations
- [ ] MLU-specific optimizations
- [ ] Ascend-specific optimizations
- [ ] DCU-specific optimizations
- [ ] GCU-specific optimizations

### Phase 3: Advanced Features
- [ ] Sparse data structure (SNode) optimization
- [ ] Automatic differentiation support
- [ ] AOT (Ahead-of-Time) compilation
- [ ] Multi-chip parallel execution

## Related Links

- FlagOS: https://github.com/flagos-ai
- FlagTree: https://github.com/flagos-ai/flagtree
- Taichi: https://github.com/taichi-dev/taichi

## License

Apache License 2.0 (same as Taichi)

## Authors

FlagOS Backend Integration for Taichi
Contributed to bridge Taichi and FlagOS ecosystems

---
**Date**: 2026-02-02
**Version**: Initial implementation (v0.1.0)
