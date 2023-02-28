import argparse
import json
from pathlib import Path
from typing import List
import taichi as ti
from taichi._ti_module.cppgen import generate_header
from taichi.aot.conventions.gfxruntime140 import GfxRuntime140

def module_cppgen(parser: argparse.ArgumentParser):
    """Generate C++ headers for Taichi modules."""
    parser.add_argument("MODOLE",
                        help="Path to the module directory.")
    parser.add_argument("-n",
                        "--namespace",
                        type=str,
                        help="C++ namespace if wanted.")
    parser.add_argument(
        "-m",
        "--module-name",
        type=str,
        help=
        "Module name to be a part of the module class. By default, it's the directory name.",
        default=None)
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        help="Output C++ header path.",
                        default="module.h")
    parser.set_defaults(func=module_cppgen_impl)

def module_cppgen_impl(a):
    module_path = a.MODOLE

    print(f"Generating C++ header for Taichi module: {Path(module_path).absolute()}")

    with open(f"{module_path}/metadata.json") as f:
        metadata_json = json.load(f)

    with open(f"{module_path}/graphs.json") as f:
        graphs_json = json.load(f)

    if a.module_name:
        module_name = a.module_name
    else:
        module_name = Path(module_path).name
        if module_name.endswith(".tcm"):
            module_name = module_name[:-4]

    out = generate_header(metadata_json, graphs_json, module_name, a.namespace)

    with open(a.output, "w") as f:
        f.write('\n'.join(out))

    print(f"Module header is saved to: {Path(a.output).absolute()}")


def module(arguments: List[str]):
    """Taichi module tools."""
    parser = argparse.ArgumentParser(prog='ti module', description=module.__doc__)
    subparsers = parser.add_subparsers(title="Taichi module manager commands", required=True)

    cppgen_parser = subparsers.add_parser('cppgen', help=module_cppgen.__doc__)
    module_cppgen(cppgen_parser)
    args = parser.parse_args(arguments)
    args.func(args)
