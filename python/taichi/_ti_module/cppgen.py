from typing import Any, List, Optional, Set

from taichi.aot.conventions.gfxruntime140 import GfxRuntime140, sr

dtype2ctype = {
    sr.DataType.f16: "half_t",
    sr.DataType.f32: "float",
    sr.DataType.f64: "double",
    sr.DataType.i8: "int8_t",
    sr.DataType.i16: "int16_t",
    sr.DataType.i32: "int32_t",
    sr.DataType.i64: "int64_t",
    sr.DataType.u8: "uint8_t",
    sr.DataType.u16: "uint16_t",
    sr.DataType.u32: "uint32_t",
    sr.DataType.u64: "uint64_t",
}


def check_arg(actual: str, expect: Any) -> List[str]:
    out = []

    expect = str(expect)
    out += [
        f"    if (value.{actual} != {expect}) {{",
        f'      ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "value.{actual} != {expect}");',
        "      return *this;",
        "    }",
    ]

    return out


def get_arg_dst(i: int, is_named: bool) -> str:
    if is_named:
        return f"args_[{i}].argument"
    return f"args_[{i}]"


def generate_scalar_assign(cls_name: str, i: int, arg_name: str, arg: sr.ArgumentScalar, is_named: bool) -> List[str]:
    ctype = dtype2ctype[arg.dtype]

    out = []

    out += [
        f"  {cls_name} &set_{arg_name}({ctype} value) {{",
    ]

    if is_named:
        out += [
            f'    args_[{i}].name = "{arg_name}";',
        ]
    if ctype == "float":
        out += [
            f"    {get_arg_dst(i, is_named)}.type = TI_ARGUMENT_TYPE_F32;",
            f"    {get_arg_dst(i, is_named)}.value.f32 = value;",
        ]
    elif ctype == "int32_t":
        out += [
            f"    {get_arg_dst(i, is_named)}.type = TI_ARGUMENT_TYPE_I32;",
            f"    {get_arg_dst(i, is_named)}.value.i32 = value;",
        ]
    else:
        out += [
            f"    {get_arg_dst(i, is_named)}.type = TI_ARGUMENT_TYPE_SCALAR;",
            f"    {get_arg_dst(i, is_named)}.value.scalar.type = TI_DATA_TYPE_{arg.dtype.name.upper()};",
            f"    *(({ctype}*)(&{get_arg_dst(i, is_named)}.value.scalar.value)) = value;",
        ]
        assert False, f"{ctype} is not a supported scalar type."

    out += [
        "    return *this;",
        "  }",
    ]
    return out


def generate_ndarray_assign(cls_name: str, i: int, arg_name: str, arg: sr.ArgumentNdArray, is_named: bool) -> List[str]:
    out = []

    out += [
        f"  {cls_name} &set_{arg_name}(const TiNdArray &value) {{",
    ]

    out += check_arg("elem_type", f"TI_DATA_TYPE_{arg.dtype.name.upper()}")
    out += check_arg("shape.dim_count", arg.ndim)
    assert len(arg.element_shape) <= 16
    out += check_arg("elem_shape.dim_count", len(arg.element_shape))
    for j, dim in enumerate(arg.element_shape):
        out += check_arg(f"elem_shape.dims[{j}]", dim)

    if is_named:
        out += [
            f'    args_[{i}].name = "{arg_name}";',
        ]
    out += [
        f"    {get_arg_dst(i, is_named)}.type = TI_ARGUMENT_TYPE_NDARRAY;",
        f"    {get_arg_dst(i, is_named)}.value.ndarray = value;",
        "    return *this;",
        "  }",
    ]
    return out


def generate_texture_assign(
    cls_name: str,
    i: int,
    arg_name: str,
    arg: sr.ArgumentTexture | sr.ArgumentRwTexture,
    is_named: bool,
) -> List[str]:
    out = []

    out += [
        f"  {cls_name} &set_{arg_name}(const TiTexture &value) {{",
    ]

    assert arg.ndim in [1, 2, 3]
    out += check_arg("dimension", f"TI_IMAGE_DIMENSION_{arg.ndim}D")
    if isinstance(arg, sr.ArgumentRwTexture):
        out += check_arg("format", f"TI_FORMAT_{arg.fmt.name.upper()}")

    if is_named:
        out += [
            f'    args_[{i}].name = "{arg_name}";',
        ]
    out += [
        f"    {get_arg_dst(i, is_named)}.type = TI_ARGUMENT_TYPE_TEXTURE;",
        f"    {get_arg_dst(i, is_named)}.value.texture = value;",
        "    return *this;",
        "  }",
    ]
    return out


def generate_kernel_args_builder(kernel: sr.Kernel) -> List[str]:
    out = []

    out += [
        f"struct Kernel_{kernel.name} : public ti::Kernel {{",
        f"  explicit Kernel_{kernel.name}(TiRuntime runtime, TiKernel kernel) :",
        "    ti::Kernel(runtime, kernel) {",
        f"    args_.resize({len(kernel.context.args)});",
        "  }",
        "",
    ]

    cls_name = f"Kernel_{kernel.name}"
    for i, arg in enumerate(kernel.context.args):
        arg_name = arg.name if arg.name else f"arg{i}"
        if isinstance(arg, sr.ArgumentScalar):
            out += generate_scalar_assign(cls_name, i, arg_name, arg, False)
        elif isinstance(arg, sr.ArgumentNdArray):
            out += generate_ndarray_assign(cls_name, i, arg_name, arg, False)
        elif isinstance(arg, (sr.ArgumentTexture, sr.ArgumentRwTexture)):
            out += generate_texture_assign(cls_name, i, arg_name, arg, False)
        else:
            assert False
        out += [""]

    out += [
        "};",
        "",
    ]
    return out


def generate_graph_args_builder(graph: sr.Graph) -> List[str]:
    out = []

    out += [
        f"struct ComputeGraph_{graph.name} : public ti::ComputeGraph {{",
        f"  explicit ComputeGraph_{graph.name}(TiRuntime runtime, TiComputeGraph graph) :",
        "    ti::ComputeGraph(runtime, graph) {",
        f"    args_.resize({len(graph.args)});",
        "  }",
        "",
    ]

    cls_name = f"ComputeGraph_{graph.name}"
    for i, arg in enumerate(graph.args):
        arg_name = arg.name
        if isinstance(arg.arg, sr.ArgumentScalar):
            out += generate_scalar_assign(cls_name, i, arg_name, arg.arg, True)
        elif isinstance(arg.arg, sr.ArgumentNdArray):
            out += generate_ndarray_assign(cls_name, i, arg_name, arg.arg, True)
        elif isinstance(arg.arg, (sr.ArgumentTexture, sr.ArgumentRwTexture)):
            out += generate_texture_assign(cls_name, i, arg_name, arg.arg, True)
        else:
            assert False
        out += [""]

    out += [
        "};",
        "",
    ]
    return out


def generate_module_content_repr(m: GfxRuntime140, module_name: str, cgraph_kernel_names: Set[str]) -> List[str]:
    out = []

    if module_name:
        module_name = f"AotModule_{module_name}"
    else:
        module_name = "AotModule"

    out += [
        f"struct {module_name} : public ti::AotModule {{",
        f"  explicit {module_name}(TiRuntime runtime, TiAotModule aot_module, bool should_destroy = true) :",
        "    ti::AotModule(runtime, aot_module, should_destroy) {}",
        "",
        f"  static {module_name} load(TiRuntime runtime, const char *path) {{",
        "    TiAotModule aot_module = ti_load_aot_module(runtime, path);",
        f"    return {module_name}(runtime, aot_module, true);",
        "  }",
        f"  static {module_name} load(TiRuntime runtime, const std::string &path) {{",
        f"    return {module_name}::load(runtime, path.c_str());",
        "  }",
        f"  static {module_name} create(TiRuntime runtime, const void *tcm, size_t size) {{",
        "    TiAotModule aot_module = ti_create_aot_module(runtime, tcm, size);",
        f"    return {module_name}(runtime, aot_module, true);",
        "  }",
        f"  static {module_name} create(TiRuntime runtime, const std::vector<uint8_t> &tcm) {{",
        f"    return {module_name}::create(runtime, tcm.data(), tcm.size());",
        "  }",
        "",
    ]
    for kernel in m.metadata.kernels.values():
        if kernel.name in cgraph_kernel_names:
            continue
        out += [
            f"  Kernel_{kernel.name} get_kernel_{kernel.name}() const {{",
            f'    return Kernel_{kernel.name}(runtime_, ti_get_aot_module_kernel(aot_module(), "{kernel.name}"));',
            "  }",
        ]
    for graph in m.graphs:
        out += [
            f"  ComputeGraph_{graph.name} get_compute_graph_{graph.name}() const {{",
            f'    return ComputeGraph_{graph.name}(runtime_, ti_get_aot_module_compute_graph(aot_module(), "{graph.name}"));',
            "  }",
        ]
    out += [
        "};",
        "",
    ]
    return out


def generate_module_content(m: GfxRuntime140, module_name: str) -> List[str]:
    # This has all kernels including all the ones launched by compute graphs.
    cgraph_kernel_names = set(dispatch.kernel.name for graph in m.graphs for dispatch in graph.dispatches)

    out = []
    for kernel in m.metadata.kernels.values():
        if kernel.name in cgraph_kernel_names:
            continue
        out += generate_kernel_args_builder(kernel)

    for graph in m.graphs:
        out += generate_graph_args_builder(graph)

    out += generate_module_content_repr(m, module_name, cgraph_kernel_names)

    return out


def generate_header(m: GfxRuntime140, module_name: str, namespace: str, tcm: Optional[bytes]) -> List[str]:
    out = []

    out += [
        "// THIS IS A GENERATED HEADER; PLEASE DO NOT MODIFY.",
        "#pragma once",
        "#include <vector>",
        "#include <string>",
        "#include <taichi/cpp/taichi.hpp>",
        "",
    ]

    if namespace:
        out += [
            f"namespace {namespace} {{",
            "",
        ]

    if tcm is not None:
        tcm_bytes = [x for x in tcm]

        out += [
            f"static const uint8_t {module_name}_tcm[{len(tcm_bytes)}] = {{",
        ]

        out += [f"  {', '.join(str(x) for x in tcm_bytes[i:i + 8])}," for i in range(0, len(tcm_bytes), 8)]

        out += [
            "};",
            "",
            f"static const size_t {module_name}_tcm_size = {len(tcm_bytes)};",
            "",
        ]

    out += generate_module_content(m, module_name)

    if namespace:
        out += [f"}} // namespace {namespace}", ""]

    return out
