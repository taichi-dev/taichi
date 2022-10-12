from taichi_json import (Alias, BitField, BuiltInType, Definition, EntryBase,
                         Enumeration, Field, Function, Handle, Module,
                         Structure, Union)

from os import system
import re


def get_type_name(x: EntryBase):
    ty = type(x)
    if ty in [BuiltInType]:
        return x.type_name
    elif ty in [Alias, Handle, Enumeration, Structure, Union]:
        return x.name.upper_camel_case
    elif ty in [BitField]:
        return x.name.extend('flags').upper_camel_case
    else:
        raise RuntimeError(f"'{x.id}' is not a type")


def get_field(x: Field):
    # `count` is an integer so it's a static array.
    is_dyn_array = x.count and not isinstance(x.count, int)

    is_ptr = x.by_ref or x.by_mut or is_dyn_array
    const_q = "const " if not x.by_mut else ""
    type_name = get_type_name(x.type)

    if is_ptr:
        return f"{const_q}{type_name}* {x.name}"
    elif x.count:
        return f"{type_name} {x.name}[{x.count}]"
    else:
        return f"{type_name} {x.name}"


def get_declr(x: EntryBase):
    ty = type(x)
    if ty is BuiltInType:
        return ""

    elif ty is Alias:
        return f"typedef {get_type_name(x.alias_of)} {get_type_name(x)};"

    elif ty is Definition:
        return f"#define {x.name.screaming_snake_case} {x.value}"

    elif ty is Handle:
        return f"typedef struct {get_type_name(x)}_t* {get_type_name(x)};"

    elif ty is Enumeration:
        out = ["typedef enum " + get_type_name(x) + " {"]
        for name, value in x.cases.items():
            out += [f"  {name.screaming_snake_case} = {value},"]
        out += [
            f"  {x.name.extend('max_enum').screaming_snake_case} = 0xffffffff,"
        ]
        out += ["} " + get_type_name(x) + ";"]
        return '\n'.join(out)

    elif ty is BitField:
        bit_type_name = x.name.extend('flag_bits').upper_camel_case
        out = ["typedef enum " + bit_type_name + " {"]
        for name, value in x.bits.items():
            out += [
                f"  {name.extend('bit').screaming_snake_case} = 1 << {value},"
            ]
        out += ["} " + bit_type_name + ";"]
        out += [f"typedef TiFlags {get_type_name(x)};"]
        return '\n'.join(out)

    elif ty is Structure:
        out = ["typedef struct " + get_type_name(x) + " {"]
        for field in x.fields:
            out += [f"  {get_field(field)};"]
        out += ["} " + get_type_name(x) + ";"]
        return '\n'.join(out)

    elif ty is Union:
        out = ["typedef union " + get_type_name(x) + " {"]
        for variant in x.variants:
            out += [f"  {get_field(variant)};"]
        out += ["} " + get_type_name(x) + ";"]
        return '\n'.join(out)

    elif ty is Function:
        return_value_type = "void" if x.return_value_type == None else get_type_name(
            x.return_value_type)
        out = [
            "TI_DLL_EXPORT " + return_value_type + " TI_API_CALL " +
            x.name.snake_case + "("
        ]
        if x.params:
            out += [',\n'.join(f"  {get_field(param)}" for param in x.params)]
        out += [");"]
        return '\n'.join(out)

    else:
        raise RuntimeError(f"'{x.id}' doesn't need declaration")


def get_human_readable_name(x: EntryBase):
    ty = type(x)
    if ty is BuiltInType:
        return ""

    elif ty is Alias:
        return f"{get_type_name(x)}"

    elif ty is Definition:
        return f"{x.name.screaming_snake_case}"

    elif isinstance(x, (Handle, Enumeration, BitField, Structure, Union)):
        return f"{get_type_name(x)}"

    elif ty is Function:
        return f"{x.name.snake_case}"

    else:
        raise RuntimeError(f"'{x.id}' doesn't have a human readable name")


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
    SYM_PATTERN = r"\`(\w+\.\w+(?:\.\w+)?)\`"
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


def print_module_header(module: Module):
    out = []
    if module.doc is not None:
        out += [
            f"// {resolve_inline_symbols_to_names(module, x)}" for x in module.doc.module_doc]
        # Remove the trailing `## API References`.
        del out[-1]
    out += ["#pragma once", ""]

    for (name, value) in module.default_definitions:
        out += [
            f"#ifndef {name}",
            f"#define {name} {value}",
            f"#endif // {name}",
            "",
        ]

    out += [
        "#include <taichi/taichi.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif // __cplusplus",
        "",
    ]

    for x in module.declr_reg:
        declr = module.declr_reg.resolve(x)
        out += [
            "",
            f"// {get_title(declr)}",
        ]
        if module.doc is not None:
            out += [
                f"// {resolve_inline_symbols_to_names(module, y)}" for y in module.doc.api_refs[x]]
        out += [get_declr(declr)]

    out += [
        "",
        "#ifdef __cplusplus",
        '} // extern "C"',
        "#endif // __cplusplus",
        "",
    ]

    return '\n'.join(out)


def generate_module_header(module):
    if module.is_built_in:
        return

    print(f"processing module '{module.name}'")
    path = f"c_api/include/{module.name}"
    with open(path, "w") as f:
        f.write(print_module_header(module))

    system(f"clang-format {path} -i")


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
        BuiltInType("GLuint", "GLuint"),
    }

    for module in Module.load_all(builtin_tys):
        generate_module_header(module)
