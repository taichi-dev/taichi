import argparse
import runpy
from pathlib import Path
from typing import List

from taichi._ti_module.cppgen import generate_header
from taichi.aot._export import _aot_kernels
from taichi.aot.conventions.gfxruntime140 import GfxRuntime140
from taichi.aot.module import Module
from taichi.types.ndarray_type import NdarrayType
from taichi.types.primitive_types import integer_type_ids, real_type_ids
from taichi.types.texture_type import RWTextureType, TextureType

import taichi


def module_cppgen(parser: argparse.ArgumentParser):
    """Generate C++ headers for Taichi modules."""
    parser.add_argument("MODOLE", help="Path to the module directory.")
    parser.add_argument("-n", "--namespace", type=str, help="C++ namespace if wanted.")
    parser.add_argument(
        "-m",
        "--module-name",
        type=str,
        help="Module name to be a part of the module class. By default, it's the directory name.",
        default=None,
    )
    parser.add_argument("-o", "--output", type=str, help="Output C++ header path.", default="module.h")
    parser.add_argument(
        "--bin2c",
        help="Save the entire TCM archive to an in-memory buffer. This flag is ignored if the module is not a TCM archive",
        action="store_true",
    )
    parser.set_defaults(func=module_cppgen_impl)


def module_cppgen_impl(a):
    module_path = a.MODOLE

    print(f"Generating C++ header for Taichi module: {Path(module_path).absolute()}")

    tcm = None
    if a.bin2c and module_path.endswith(".tcm"):
        with open(module_path, "rb") as f:
            tcm = f.read()

    if a.module_name:
        module_name = a.module_name
    else:
        module_name = Path(module_path).name
        if module_name.endswith(".tcm"):
            module_name = module_name[:-4]

    m = GfxRuntime140.from_module(module_path)

    out = generate_header(m, module_name, a.namespace, tcm)

    with open(a.output, "w") as f:
        f.write("\n".join(out))

    print(f"Module header is saved to: {Path(a.output).absolute()}")


def module_build(parser: argparse.ArgumentParser):
    """Build Taichi modules from python scripts."""
    parser.add_argument("SOURCE", help="Path to the Taichi program source (Python script).")
    parser.add_argument("-o", "--output", type=str, help="Output module path.", default=None)
    parser.set_defaults(func=module_build_impl)


def module_build_impl(a):
    source_path = a.SOURCE
    module_path = a.output

    source_path = Path(source_path)
    assert source_path.name.endswith(".py"), "Source must be a Python script."
    if module_path is None:
        module_path = f"{source_path.name[:-3]}.tcm"
    module_path = Path(module_path)

    print(f"Building Taichi module: {source_path}")
    print()

    d = runpy.run_path(str(source_path), run_name="__main__")
    print()

    required_caps = d["REQUIRED_CAPS"] if "REQUIRED_CAPS" in d else []
    assert isinstance(required_caps, list), "REQUIRED_CAPS must be a list."

    if required_caps:
        print("Module requires the following capabilities:")
        for cap in required_caps:
            print(f"  - {cap}")
        print()

    m = Module(caps=required_caps)
    for record in _aot_kernels:
        print("Added kernel:", record.name)
        template_args = None
        if record.template_types:
            print("  Template types:")
            template_args = {}
            for k, v in record.template_types.items():
                print(f"    - {k}: {v}")
                # TODO: (penguinliong) Remove this hack. It's not properly
                # working with unusual numeric types like f16 or i64.
                if isinstance(v, int) or id(v) in integer_type_ids:
                    value = 0
                elif isinstance(v, float) or id(v) in real_type_ids:
                    value = 0.0
                elif isinstance(v, NdarrayType):
                    if v.ndim is None or v.ndim <= 0:
                        raise ValueError("Ndarray template type must specify a non-zero dimension.")
                    value = taichi.ndarray(v.dtype, (1,) * v.ndim)
                elif isinstance(v, TextureType):
                    value = taichi.Texture(taichi.Format.rgba8, (4,) * v.num_dimensions)
                elif isinstance(v, RWTextureType):
                    value = taichi.Texture(v.fmt, (4,) * v.num_dimensions)
                else:
                    raise ValueError(f"Unsupported template type: {type(v)}")
                template_args[k] = value
        m.add_kernel(record.kernel, template_args)
    print()

    if module_path.name.endswith(".tcm"):
        m.archive(str(module_path))
    else:
        m.save(str(module_path))

    print(f"Module is archive to: {module_path}")
    print()


def _main(arguments: List[str]):
    """Taichi module tools."""
    parser = argparse.ArgumentParser(prog="ti module", description=_main.__doc__)
    subparsers = parser.add_subparsers(title="Taichi module manager commands", required=True)

    cppgen_parser = subparsers.add_parser("cppgen", help=module_cppgen.__doc__)
    build_parser = subparsers.add_parser("build", help=module_build.__doc__)
    module_cppgen(cppgen_parser)
    module_build(build_parser)
    args = parser.parse_args(arguments)
    args.func(args)
