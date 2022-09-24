import re
from collections import defaultdict
from pathlib import Path

from generate_c_api import get_declr, get_human_readable_name
from taichi_json import (Alias, BitField, BuiltInType, Definition, EntryBase,
                         Enumeration, Field, Function, Handle, Module,
                         Structure, Union)

SYM_PATTERN = r"\`(\w+\.\w+(?:\.\w+)?)\`"


def get_title(x: EntryBase):
    if isinstance(x, BuiltInType):
        return ""

    extra = ""
    if isinstance(x, Function) and x.is_device_command:
        extra += " (Device Command)"

    if isinstance(x, (Alias, Definition, Handle, Enumeration, BitField,
                      Structure, Union, Function)):
        return f"{type(x).__name__} `{get_human_readable_name(x)}`" + extra
    else:
        raise RuntimeError(f"'{x.id}' doesn't need title")


def get_human_readable_field_name(x: EntryBase, field_name: str):
    out = None
    if isinstance(x, Enumeration):
        out = x.name.extend(field_name).screaming_snake_case
    elif isinstance(x, BitField):
        out = x.name.extend(field_name).extend('bit').screaming_snake_case
    elif isinstance(x, Structure):
        for field in x.fields:
            if str(field.name) == field_name:
                out = str(field.name)
                break
    elif isinstance(x, Union):
        for field in x.variants:
            if str(field.name) == field_name:
                out = str(field.name)
                break
    elif isinstance(x, Function):
        for field in x.params:
            if str(field.name) == field_name:
                out = str(field.name)
                break
    return out


def resolve_symbol_to_name(module: Module, id: str):
    """Returns the resolved symbol and its hyperlink (if available)"""
    try:
        ifirst_dot = id.index('.')
    except ValueError:
        return None

    field_name = ""
    try:
        isecond_dot = id.index('.', ifirst_dot + 1)
        field_name = id[isecond_dot + 1:]
        id = id[:isecond_dot]
    except ValueError:
        pass

    out = module.declr_reg.resolve(id)
    href = None

    try:
        if field_name:
            out = get_human_readable_field_name(out, field_name)
        else:
            href = "#" + get_title(out).lower().replace(' ', '-').replace(
                '`', '').replace('(', '').replace(')', '')
            out = get_human_readable_name(out)
    except:
        print(f"WARNING: Unable to resolve symbol {id}")
        out = id

    return out, href


def resolve_inline_symbols_to_names(module: Module, line: str):
    matches = re.findall(SYM_PATTERN, line)

    replacements = {}
    for m in matches:
        id = str(m)
        replacements[id] = resolve_symbol_to_name(module, id)

    for old, (new, href) in replacements.items():
        if new is None:
            print(f"WARNING: Unresolved inline symbol `{old}`")
        else:
            if href is None:
                new = f"`{new}`"
            else:
                new = f"[`{new}`]({href})"
            line = line.replace(f"`{old}`", new)
    return line


def print_module_doc(module: Module, templ):
    out = []

    for i in range(len(templ)):
        line = templ[i].strip()
        line = resolve_inline_symbols_to_names(module, line)
        out += [line]
        if line.startswith("## API Reference"):
            break

    out += [""]

    cur_sym = None
    documented_syms = defaultdict(list)
    for line in templ[i:]:
        line = line.strip()
        if re.match(SYM_PATTERN, line):
            cur_sym = line[1:-1]
            continue
        documented_syms[cur_sym] += [
            resolve_inline_symbols_to_names(module, line)
        ]

    is_first = True
    for x in module.declr_reg:
        declr = module.declr_reg.resolve(x)

        if is_first:
            is_first = False
        else:
            out += ["---"]

        out += [
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

    out += [""]

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
    path = f"docs/lang/articles/c-api/{module.name[7:-2]}.md"
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
        BuiltInType("VkImage", "VkImage"),
        BuiltInType("VkImageType", "VkImageType"),
        BuiltInType("VkFormat", "VkFormat"),
        BuiltInType("VkExtent3D", "VkExtent3D"),
        BuiltInType("VkSampleCountFlagBits", "VkSampleCountFlagBits"),
        BuiltInType("VkImageTiling", "VkImageTiling"),
        BuiltInType("VkImageLayout", "VkImageLayout"),
        BuiltInType("VkImageUsageFlags", "VkImageUsageFlags"),
        BuiltInType("VkImageViewType", "VkImageViewType"),
        BuiltInType("PFN_vkGetInstanceProcAddr", "PFN_vkGetInstanceProcAddr"),
        BuiltInType("char", "char"),
    }

    for module in Module.load_all(builtin_tys):
        generate_module_header(module)
