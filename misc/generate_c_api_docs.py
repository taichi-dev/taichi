import re
from collections import defaultdict
from pathlib import Path

from generate_c_api import get_declr, get_field, get_type_name
from taichi_json import (Alias, BitField, BuiltInType, Definition, EntryBase,
                         Enumeration, Field, Function, Handle, Module,
                         Structure, Union)


def get_title(x: EntryBase):
    ty = type(x)
    if ty is BuiltInType:
        return ""

    elif ty is Alias:
        return f"Alias `{get_type_name(x)}`"

    elif ty is Definition:
        return f"Definition `{x.name.screaming_snake_case}`"

    elif ty is Handle:
        return f"Handle `{get_type_name(x)}`"

    elif ty is Enumeration:
        return f"Enumeration `{get_type_name(x)}`"

    elif ty is BitField:
        return f"Bit Field `{get_type_name(x)}`"

    elif ty is Structure:
        return f"Structure `{get_type_name(x)}`"

    elif ty is Union:
        return f"Union `{get_type_name(x)}`"

    elif ty is Function:
        extra = ""
        if x.is_device_command:
            extra += " (Device Command)"
        return f"Function `{x.name.snake_case}`" + extra

    else:
        raise RuntimeError(f"'{x.id}' doesn't need title")


def print_module_doc(module: Module, templ):
    out = []

    for i in range(len(templ)):
        line = templ[i]
        out += [line.strip()]
        if line.startswith("## Declarations"):
            break

    out += [""]

    cur_sym = None
    documented_syms = defaultdict(list)
    for line in templ[i:]:
        line = line.strip()
        if re.match(r"\`\w+\.\w+\`", line):
            cur_sym = line[1:-1]
            continue
        documented_syms[cur_sym] += [line]

    for x in module.declr_reg:
        declr = module.declr_reg.resolve(x)

        out += [
            "---",
            f"### {get_title(declr)}",
            "",
            "```c",
            f"// {x}",
            get_declr(declr),
            "```",
        ]

        if x in documented_syms:
            out += documented_syms[x]
        else:
            print(f"WARNING: `{x}` is not documented")

    return '\n'.join(out)


def generate_module_header(module):
    if module.is_built_in:
        return

    templ_path = f"c_api/docs/{module.name}.md"
    templ = None
    if Path(templ_path).exists():
        with open(templ_path) as f:
            templ = f.readlines()
    else:
        print(
            f"ignored {templ_path} because the documentation template cannot be found"
        )
        return

    print(f"processing module '{module.name}'")
    path = f"docs/c_api/{module.name}.md"
    with open(path, "w") as f:
        f.write(print_module_doc(module, templ))

    #system(f"clang-format {path} -i")


if __name__ == "__main__":
    builtin_tys = {
        BuiltInType("uint64_t", "uint64_t"),
        BuiltInType("int64_t", "int64_t"),
        BuiltInType("uint32_t", "uint32_t"),
        BuiltInType("int32_t", "int32_t"),
        BuiltInType("float", "float"),
        BuiltInType("const char*", "const char*"),
        BuiltInType("const char**", "const char**"),
        BuiltInType("void*", "void*"),
        BuiltInType("const void*", "const void*"),
        BuiltInType("VkInstance", "VkInstance"),
        BuiltInType("VkPhysicalDevice", "VkPhysicalDevice"),
        BuiltInType("VkDevice", "VkDevice"),
        BuiltInType("VkQueue", "VkQueue"),
        BuiltInType("VkBuffer", "VkBuffer"),
        BuiltInType("VkBufferUsageFlags", "VkBufferUsageFlags"),
        BuiltInType("VkEvent", "VkEvent"),
    }

    for module in Module.load_all(builtin_tys):
        generate_module_header(module)
