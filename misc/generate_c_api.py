from taichi_json import (Alias, BitField, BuiltInType, Definition, EntryBase,
                         Enumeration, Field, Function, Handle, Module,
                         Structure, Union)

#from os import system


def get_type_name(x: EntryBase):
    ty = type(x)
    if ty in [BuiltInType]:
        return x.type_name
    elif ty in [Alias, Handle, Enumeration, Structure, Union]:
        return x.name.upper_camel_case
    elif ty in [BitField]:
        return x.name.extend('flag_bits').upper_camel_case
    else:
        raise RuntimeError(f"'{x.id}' is not a type")


def get_field(x: Field):
    # `count` is an integer so it's a static array.
    is_dyn_array = x.count and not isinstance(x.count, int)

    is_ptr = x.by_ref or x.by_mut or is_dyn_array
    const_q = "const" if not x.by_mut else ""
    type_name = get_type_name(x.type)

    if is_ptr:
        return f"{const_q} {type_name}* {x.name}"
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
        out = ["typedef enum " + get_type_name(x) + " {"]
        for name, value in x.bits.items():
            out += [
                f"  {name.extend('bit').screaming_snake_case} = 1 << {value},"
            ]
        out += ["} " + get_type_name(x) + ";"]
        out += [f"typedef TiFlags {x.name.extend('flags').upper_camel_case};"]
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


def print_module_header(module):
    out = ["#pragma once"]

    for x in module.required_modules:
        out += [f"#include <{x}>"]

    out += [
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif // __cplusplus",
        "",
    ]

    for x in module.declr_reg:
        out += [
            "",
            f"// {x}",
            get_declr(module.declr_reg.resolve(x)),
        ]

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
        BuiltInType("char", "char"),
    }

    for module in Module.load_all(builtin_tys):
        generate_module_header(module)
