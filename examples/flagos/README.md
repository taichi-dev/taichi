# Taichi FlagOS Backend Examples

This directory contains example programs demonstrating the Taichi FlagOS backend for various AI chips.

## Prerequisites

1. **FlagOS SDK**: Install FlagOS SDK (version >= 1.5)
2. **FlagTree Compiler**: Install FlagTree compiler (version >= 0.8)
3. **Taichi**: Build Taichi from source with FlagOS support

## Building Taichi with FlagOS Support

```bash
# Clone Taichi repository
git clone https://github.com/taichi-dev/taichi.git
cd taichi

# Build with FlagOS support
mkdir build && cd build
cmake .. -DTI_WITH_FLAGOS=ON \
         -DTI_WITH_LLVM=ON \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install
pip install -e .
```

## Supported AI Chips

| Chip | Vendor | Status |
|------|--------|--------|
| MLU370 | Cambricon | Planned |
| MLU590 | Cambricon | Planned |
| Ascend910 | Huawei | Planned |
| Ascend310 | Huawei | Planned |
| DCU | Hygon | Planned |
| GCU | Enflame | Planned |
| Generic | - | Available (stub) |

## Usage

### Method 1: Environment Variable

```bash
export TI_FLAGOS_CHIP=mlu370
python fractal_flagos.py
```

### Method 2: Programmatic Configuration

```python
import taichi as ti

ti.init(arch=ti.flagos, flagos_chip="mlu370")
```

## Examples

### 1. Julia Set Fractal (`fractal_flagos.py`)

A classic parallel computation example that generates a Julia set fractal.

```bash
python fractal_flagos.py
```

### 2. Matrix Multiplication (`matmul_flagos.py`)

Benchmarks matrix multiplication performance on FlagOS-supported chips.

```bash
python matmul_flagos.py
```

## Troubleshooting

### FlagOS backend not available

If you get `AttributeError: module 'taichi' has no attribute 'flagos'`, it means:

1. Taichi was not built with FlagOS support. Rebuild with `-DTI_WITH_FLAGOS=ON`.
2. The FlagOS SDK was not found during build. Check CMake output for warnings.

### Chip not supported

If you get an error about unsupported chip:

1. Check that the chip name is correct (case-sensitive)
2. Verify that FlagOS SDK supports your target chip
3. Use `generic` as a fallback for testing

### Performance issues

1. Enable kernel profiling: `ti.init(..., kernel_profiler=True)`
2. Check chip-specific optimization flags in FlagOS documentation
3. Adjust block/grid dimensions for your specific chip

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Taichi Frontend                    │
│                   (Python API)                        │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                   Taichi Compiler                     │
│            (IR Transformation & Optimization)         │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                 FlagOS Code Generator                 │
│              (LLVM IR for FlagTree)                   │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                   FlagTree Compiler                   │
│         (MLIR/LLVM to Target Chip Code)              │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              AI Chip (MLU/Ascend/DCU/GCU)             │
└─────────────────────────────────────────────────────┘
```

## Contributing

Contributions to improve FlagOS backend support are welcome! Please:

1. Check existing issues and PRs
2. Follow Taichi's contribution guidelines
3. Add tests for new features
4. Update documentation

## Resources

- [FlagOS GitHub](https://github.com/flagos-ai)
- [FlagTree Compiler](https://github.com/flagos-ai/flagtree)
- [Taichi Documentation](https://docs.taichi-lang.org)

## License

These examples are released under the same license as Taichi (Apache 2.0).
