from typing import Any, List

import taichi.aot.conventions.gfxruntime140.sr as sr
from taichi.aot.conventions.gfxruntime140 import GfxRuntime140

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
        f"      ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, \"value.{actual} != {expect}\");",
        f"      return *this;",
        f"    }}",
    ]

    return out


def get_arg_dst(i: int, is_named: bool) -> str:
    if is_named:
        return f"args[{i}].argument"
    else:
        return f"args[{i}]"


def generate_scalar_assign(cls_name: str, i: int, arg_name: str,
                           arg: sr.ArgumentScalar,
                           is_named: bool) -> List[str]:
    ctype = dtype2ctype[arg.dtype]

    out = []

    out += [
        f"  {cls_name} &set_{arg_name}({ctype} value) {{",
    ]

    if is_named:
        out += [
            f"    args[{i}].name = \"{arg_name}\";",
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


def generate_ndarray_assign(cls_name: str, i: int, arg_name: str,
                            arg: sr.ArgumentNdArray,
                            is_named: bool) -> List[str]:
    out = []

    out += [
        f"  {cls_name} &set_{arg_name}(const TiNdArray &value) {{",
    ]

    out += check_arg("elem_type", arg.dtype)
    out += check_arg("shape.dim_count", arg.ndim)
    assert len(arg.element_shape) <= 16
    out += check_arg("elem_shape.dim_count", len(arg.element_shape))
    for j, dim in enumerate(arg.element_shape):
        out += check_arg(f"elem_shape.dims[{j}]", dim)

    if is_named:
        out += [
            f"    args[{i}].name = \"{arg_name}\";",
        ]
    out += [
        f"    {get_arg_dst(i, is_named)}.type = TI_ARGUMENT_TYPE_NDARRAY;",
        f"    {get_arg_dst(i, is_named)}.value.ndarray = value;",
        "    return *this;",
        "  }",
    ]
    return out


def generate_texture_assign(cls_name: str, i: int, arg_name: str,
                            arg: sr.ArgumentTexture | sr.ArgumentRwTexture,
                            is_named: bool) -> List[str]:
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
            f"    args[{i}].name = \"{arg_name}\";",
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
        f"struct Kernel_{kernel.name} {{",
        "  TiRuntime runtime;",
        "  TiKernel kernel;",
        f"  TiArgument args[{len(kernel.context.args)}];",
        "",
        f"  explicit Kernel_{kernel.name}(TiRuntime runtime, TiKernel kernel) :",
        f"    runtime(runtime), kernel(kernel) {{}}",
        f"  explicit Kernel_{kernel.name}(TiRuntime runtime, TiAotModule aot_module) :",
        f"    runtime(runtime), kernel(ti_get_aot_module_kernel(aot_module, \"{kernel.name}\")) {{}}",
        "",
    ]

    cls_name = f"Kernel_{kernel.name}"
    for i, arg in enumerate(kernel.context.args):
        arg_name = f"arg_{i}"
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
        "  void launch() const {",
        f"    ti_launch_kernel(runtime, kernel, {len(kernel.context.args)}, args);",
        "  }",
        "};",
        "",
    ]
    return out


def generate_graph_args_builder(graph: sr.Graph) -> List[str]:
    out = []

    out += [
        f"struct Graph_{graph.name} {{",
        "  TiRuntime runtime;",
        "  TiComputeGraph graph;",
        f"  TiNamedArgument args[{len(graph.args)}];",
        "",
        f"  explicit Graph_{graph.name}(TiRuntime runtime, TiComputeGraph graph) :",
        f"    runtime(runtime), graph(graph) {{}}",
        f"  explicit Graph_{graph.name}(TiRuntime runtime, TiAotModule aot_module) :",
        f"    runtime(runtime), graph(ti_get_aot_module_compute_graph(aot_module, \"{graph.name}\")) {{}}",
        "",
    ]

    cls_name = f"Graph_{graph.name}"
    for i, arg in enumerate(graph.args):
        arg_name = arg.name
        if isinstance(arg.arg, sr.ArgumentScalar):
            out += generate_scalar_assign(cls_name, i, arg_name, arg.arg, True)
        elif isinstance(arg.arg, sr.ArgumentNdArray):
            out += generate_ndarray_assign(cls_name, i, arg_name, arg.arg,
                                           True)
        elif isinstance(arg.arg, (sr.ArgumentTexture, sr.ArgumentRwTexture)):
            out += generate_texture_assign(cls_name, i, arg_name, arg.arg,
                                           True)
        else:
            assert False
        out += [""]

    out += [
        "  void launch() const {",
        f"    ti_launch_compute_graph(runtime, graph, {len(graph.args)}, args);",
        "  }",
        "};",
        "",
    ]
    return out


def generate_module_content_repr(m: GfxRuntime140,
                                 module_name: str) -> List[str]:
    out = []

    if module_name:
        out += [
            f"struct Module_{module_name} {{",
        ]
    else:
        out += [
            f"struct Module {{",
        ]

    out += [
        "  TiRuntime runtime;",
        "  TiAotModule aot_module;",
        "  bool should_destroy;",
        "",
        f"  explicit Module(TiRuntime runtime, TiAotModule aot_module, bool should_destroy = true) :",
        f"    runtime(runtime), aot_module(aot_module), should_destroy(should_destroy) {{}}",
        "  ~Module() {",
        "    if (should_destroy) {",
        "      ti_destroy_aot_module(aot_module);",
        "    }",
        "  }",
        "",
    ]
    for kernel in m.metadata.kernels.values():
        out += [
            f"  Kernel_{kernel.name} get_kernel_{kernel.name}() const {{",
            f"    return Kernel_{kernel.name}(runtime, aot_module);",
            "  }",
        ]
    for graph in m.graphs:
        out += [
            f"  Graph_{graph.name} get_graph_{graph.name}() const {{",
            f"    return Graph_{graph.name}(runtime, aot_module);",
            "  }",
        ]
    out += [
        "};",
        "",
    ]
    return out


def generate_module_content(m: GfxRuntime140, module_name: str) -> List[str]:
    # This has all kernels including all the ones launched by compute graphs.
    cgraph_kernel_names = set(dispatch.kernel.name for graph in m.graphs
                              for dispatch in graph.dispatches)

    out = []
    for kernel in m.metadata.kernels.values():
        if kernel.name in cgraph_kernel_names:
            continue
        out += generate_kernel_args_builder(kernel)

    for graph in m.graphs:
        out += generate_graph_args_builder(graph)

    out += generate_module_content_repr(m, module_name)

    return out


def generate_header(metadata_json: str, graphs_json: str, module_name: str,
                    namespace: str) -> List[str]:
    out = []

    m = GfxRuntime140(metadata_json, graphs_json)

    out += [
        "// THIS IS A GENERATED HEADER; PLEASE DO NOT MODIFY.",
        "#pragma once",
        "#include <taichi/taichi.h>",
        "",
    ]

    if namespace:
        out += [
            f"namespace {namespace} {{",
            "",
        ]

    out += generate_module_content(m, module_name)

    if namespace:
        out += [f"}} // namespace {namespace}", ""]

    return out
