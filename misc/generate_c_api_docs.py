from generate_c_api import (get_declr, get_title,
                            resolve_inline_symbols_to_names)
from taichi_json import BuiltInType, Module


def print_module_doc(module: Module):
    out = ["---"]
    out += module.doc.markdown_metadata
    out += ["---", ""]
    for line in module.doc.module_doc:
        out += [resolve_inline_symbols_to_names(module, line)]
    out += [""]

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
            get_declr(module, declr),
            "```",
        ]

        if x in module.doc.api_full_refs:
            for line in module.doc.api_full_refs[x]:
                out += [resolve_inline_symbols_to_names(module, line)]
        else:
            print(f"WARNING: `{x}` is not documented")
            out += [""]
    out += [""]

    return '\n'.join(out)


def generate_module_header(module):
    if module.is_built_in or module.doc is None:
        return

    print(f"processing module '{module.name}'")
    path = f"docs/lang/articles/c-api/{module.name[7:-2]}.md"
    with open(path, "w") as f:
        f.write(print_module_doc(module))

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
        BuiltInType("GLuint", "GLuint"),
    }

    for module in Module.load_all(builtin_tys):
        generate_module_header(module)
