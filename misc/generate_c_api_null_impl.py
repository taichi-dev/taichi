from os import system

from taichi_json import (Alias, BitField, BuiltInType, EntryBase, Enumeration,
                         Field, Function, Handle, Module, Structure, Union)


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


def get_definition(x: EntryBase):
    out = []

    ty = type(x)
    if ty is Function:
        return_value_type = "void" if x.return_value_type == None else get_type_name(
            x.return_value_type)
        out += [
            "TI_DLL_EXPORT " + return_value_type + " TI_API_CALL " +
            x.name.snake_case + "("
        ]
        if x.params:
            for i, param in enumerate(x.params):
                if i != 0:
                    out[-1] += ","
                out += [f"  {get_field(param)}"]
        out += [
            ") {",
        ]
        if return_value_type != "void":
            out += [f"return ({return_value_type})0;"]
        out += [
            "}"
            "",
        ]

    return out


def print_module_header(module: Module):
    out = []
    if module.guarded_by != None:
        out += [
            f"#ifdef {module.guarded_by}",
            "",
        ]

    for x in module.declr_reg:
        declr = module.declr_reg.resolve(x)
        out += get_definition(declr)

    if module.guarded_by != None:
        out += [
            f"#endif // {module.guarded_by}",
            "",
        ]

    return out


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
        BuiltInType("VkDeviceMemory", "VkDeviceMemory"),
    }

    out = [
        "#include <taichi/taichi.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif // __cplusplus",
        "",
    ]

    for module in Module.load_all(builtin_tys):
        if module.is_built_in:
            continue

        print(f"processing module '{module.name}'")
        out += print_module_header(module)

    out += [
        "",
        "#ifdef __cplusplus",
        '} // extern "C"',
        "#endif // __cplusplus",
        "",
    ]

    path = f"c_api/link/c_api_null_impl.cpp"
    with open(path, "w") as f:
        f.write('\n'.join(out))

    system(f"clang-format {path} -i")
