import re
from os import system

from taichi_json import (
    Alias,
    BitField,
    BuiltInType,
    Callback,
    Definition,
    EntryBase,
    Enumeration,
    Field,
    Function,
    Handle,
    Module,
    Structure,
    Union,
)


def get_type_name(x: EntryBase):
    ty = type(x)
    if ty in [BuiltInType]:
        return x.type_name
    elif ty in [Alias, Handle, Enumeration, Structure, Union, Callback]:
        return x.name.upper_camel_case
    elif ty in [BitField]:
        return x.name.extend("flags").upper_camel_case
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


def get_api_ref(module: Module, x: EntryBase) -> list:
    out = [f"// {get_title(x)}"]
    if x.since is not None:
        out[-1] += f" ({x.since})"
    if module.doc and x.id in module.doc.api_refs:
        out += [f"// {resolve_inline_symbols_to_names(module, y)}" for y in module.doc.api_refs[x.id]]
    return out


def get_api_field_ref(module: Module, x: EntryBase, field_sym: str) -> list:
    field_sym = f"{x.id}.{field_sym}"
    if module.doc and field_sym in module.doc.api_field_refs:
        return [f"  // {module.doc.api_field_refs[field_sym]}"]
    return []


def get_declr(module: Module, x: EntryBase, with_docs=False):
    out = []
    if with_docs:
        out += get_api_ref(module, x)

    ty = type(x)
    if ty is BuiltInType:
        out += [""]

    elif ty is Alias:
        out += [f"typedef {get_type_name(x.alias_of)} {get_type_name(x)};"]

    elif ty is Definition:
        out += [f"#define {x.name.screaming_snake_case} {x.value}"]

    elif ty is Handle:
        out += [f"typedef struct {get_type_name(x)}_t* {get_type_name(x)};"]

    elif ty is Enumeration:
        out += ["typedef enum " + get_type_name(x) + " {"]
        for name, value in x.cases.items():
            if with_docs:
                out += get_api_field_ref(module, x, name)
            name = x.name.extend(name).screaming_snake_case
            out += [f"  {name} = {value},"]
        out += [f"  {x.name.extend('max_enum').screaming_snake_case} = 0xffffffff,"]
        out += ["} " + get_type_name(x) + ";"]

    elif ty is BitField:
        bit_type_name = x.name.extend("flag_bits").upper_camel_case
        out += ["typedef enum " + bit_type_name + " {"]
        for name, value in x.bits.items():
            if with_docs:
                out += get_api_field_ref(module, x, name)
            name = x.name.extend(name).extend("bit").screaming_snake_case
            out += [f"  {name} = 1 << {value},"]
        out += ["} " + bit_type_name + ";"]
        out += [f"typedef TiFlags {get_type_name(x)};"]

    elif ty is Structure:
        out += ["typedef struct " + get_type_name(x) + " {"]
        for field in x.fields:
            if with_docs:
                out += get_api_field_ref(module, x, field.name)
            out += [f"  {get_field(field)};"]
        out += ["} " + get_type_name(x) + ";"]

    elif ty is Union:
        out += ["typedef union " + get_type_name(x) + " {"]
        for variant in x.variants:
            if with_docs:
                out += get_api_field_ref(module, x, variant.name)
            out += [f"  {get_field(variant)};"]
        out += ["} " + get_type_name(x) + ";"]

    elif ty is Callback:
        return_value_type = "void" if x.return_value_type == None else get_type_name(x.return_value_type)
        out += [f"typedef {return_value_type} (TI_API_CALL *{get_type_name(x)})("]
        if x.params:
            for i, param in enumerate(x.params):
                if i != 0:
                    out[-1] += ","
                if with_docs:
                    out += get_api_field_ref(module, x, param.name)
                out += [f"  {get_field(param)}"]
        out += [");"]

    elif ty is Function:
        return_value_type = "void" if x.return_value_type == None else get_type_name(x.return_value_type)
        out += ["TI_DLL_EXPORT " + return_value_type + " TI_API_CALL " + x.name.snake_case + "("]
        if x.params:
            for i, param in enumerate(x.params):
                if i != 0:
                    out[-1] += ","
                if with_docs:
                    out += get_api_field_ref(module, x, param.name)
                out += [f"  {get_field(param)}"]
        out += [");"]

    else:
        raise RuntimeError(f"'{x.id}' doesn't need declaration")

    return "\n".join(out)


def get_human_readable_name(x: EntryBase):
    ty = type(x)
    if ty is BuiltInType:
        return ""

    elif ty is Alias:
        return f"{get_type_name(x)}"

    elif ty is Definition:
        return f"{x.name.screaming_snake_case}"

    elif isinstance(x, (Handle, Enumeration, BitField, Structure, Union, Callback)):
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

    if isinstance(
        x,
        (
            Alias,
            Definition,
            Handle,
            Enumeration,
            BitField,
            Structure,
            Union,
            Callback,
            Function,
        ),
    ):
        return f"{type(x).__name__} `{get_human_readable_name(x)}`" + extra
    else:
        raise RuntimeError(f"'{x.id}' doesn't need title")


def resolve_symbol_to_name(module: Module, id: str):
    """Returns the resolved symbol and its hyperlink (if available)"""
    try:
        ifirst_dot = id.index(".")
    except ValueError:
        return None

    field_name = ""
    try:
        isecond_dot = id.index(".", ifirst_dot + 1)
        field_name = id[isecond_dot + 1 :]
        id = id[:isecond_dot]
    except ValueError:
        pass

    out = module.declr_reg.resolve(id)
    href = None

    try:
        if field_name:
            out = get_human_readable_field_name(out, field_name)
        else:
            href = "#" + get_title(out).lower().replace(" ", "-").replace("`", "").replace("(", "").replace(")", "")
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
        out = x.name.extend(field_name).extend("bit").screaming_snake_case
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
    elif isinstance(x, (Callback, Function)):
        for field in x.params:
            if str(field.name) == field_name:
                out = str(field.name)
                break
    return out


def print_module_header(module: Module):
    out = []
    if module.doc is not None:
        out += [f"// {resolve_inline_symbols_to_names(module, x)}" for x in module.doc.module_doc]
        # Remove the trailing `## API References`.
        del out[-1]
    out += ["#pragma once", ""]

    for name, value in module.default_definitions:
        out += [
            f"#ifndef {name}",
            f"#define {name} {value}",
            f"#endif // {name}",
            "",
        ]

    out += [
        "#ifndef TAICHI_H",
        '#include "taichi.h"',
        "#endif // TAICHI_H",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif // __cplusplus",
        "",
    ]

    for x in module.declr_reg:
        declr = module.declr_reg.resolve(x)
        out += ["", get_declr(module, declr, True)]

    out += [
        "",
        "#ifdef __cplusplus",
        '} // extern "C"',
        "#endif // __cplusplus",
        "",
    ]

    return "\n".join(out)


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
        BuiltInType("uint16_t", "uint16_t"),
        BuiltInType("int16_t", "int16_t"),
        BuiltInType("uint8_t", "uint8_t"),
        BuiltInType("int8_t", "int8_t"),
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
        BuiltInType("VkDeviceMemory", "VkDeviceMemory"),
        BuiltInType("GLenum", "GLenum"),
        BuiltInType("GLsizei", "GLsizei"),
        BuiltInType("GLsizeiptr", "GLsizeiptr"),
    }

    for module in Module.load_all(builtin_tys):
        generate_module_header(module)
