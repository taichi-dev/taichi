import ctypes
from ctypes.util import find_library
import numpy as np
import re

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
        return f"('{x.name}', ctypes.c_void_p) # {const_q}{type_name}*"
    elif x.count:
        return f"('{x.name}', {type_name} * {x.count})"
    else:
        return f"('{x.name}', {type_name})"


def get_param(x: Field):
    # `count` is an integer so it's a static array.
    is_dyn_array = x.count and not isinstance(x.count, int)

    is_ptr = x.by_ref or x.by_mut or is_dyn_array
    const_q = "const " if not x.by_mut else ""
    type_name = get_type_name(x.type)

    if is_ptr:
        return f"{x.name}: ctypes.c_void_p, # {const_q}{type_name}*"
    elif x.count:
        return f"{x.name}: {type_name} * {x.count},"
    else:
        return f"{x.name}: {type_name}"



def get_api_ref(module: Module, x: EntryBase) -> list:
    out = [
        '"""',
        f"{get_title(x)}"
    ]
    if x.since is not None:
        out[-1] += f" ({x.since})"
    if module.doc and x.id in module.doc.api_refs:
        out += [f"{resolve_inline_symbols_to_names(module, y)}" for y in module.doc.api_refs[x.id]]
    out += [
        '"""',
    ]
    return out


def get_api_field_ref(module: Module, x: EntryBase, field_sym: str) -> list:
    field_sym = f"{x.id}.{field_sym}"
    if module.doc and field_sym in module.doc.api_field_refs:
        return [f"# {module.doc.api_field_refs[field_sym]}"]
    return []

def get_api_param_ref(module: Module, x: EntryBase, param_name: str) -> list:
    param_sym = f"{x.id}.{param_name}"
    if module.doc and param_sym in module.doc.api_field_refs:
        return [f"{module.doc.api_field_refs[param_sym]}"]
    return []


def get_declr(module: Module, x: EntryBase, with_docs=False):
    out = []
    if with_docs and not isinstance(x, Function):
        out += get_api_ref(module, x)

    ty = type(x)
    if ty is BuiltInType:
        out += [""]

    elif ty is Alias:
        out += [f"{get_type_name(x)} = {get_type_name(x.alias_of)}"]

    elif ty is Definition:
        out += [f"{x.name.screaming_snake_case} = {x.value}"]

    elif ty is Handle:
        out += [f"{get_type_name(x)} = ctypes.c_void_p"]

    elif ty is Enumeration:
        out += [get_type_name(x) + " = ctypes.c_int32"]
        for name, value in x.cases.items():
            if with_docs:
                out += get_api_field_ref(module, x, name)
            name = x.name.extend(name).screaming_snake_case
            out += [f"{name} = {get_type_name(x)}({value})"]
        out += [f"{x.name.extend('max_enum').screaming_snake_case} = {get_type_name(x)}(0xffffffff)"]

    elif ty is BitField:
        bit_type_name = x.name.extend("flag_bits").upper_camel_case
        out += [f"{get_type_name(x)} = TiFlags"]
        out += [f"{bit_type_name} = ctypes.c_uint32"]
        for name, value in x.bits.items():
            if with_docs:
                out += get_api_field_ref(module, x, name)
            name = x.name.extend(name).extend("bit").screaming_snake_case
            out += [f"{name} = {bit_type_name}(1 << {value}),"]

    elif ty is Structure:
        out += [
            "class " + get_type_name(x) + "(ctypes.Structure): pass",
            f"{get_type_name(x)}._fields_ = [",
        ]
        for field in x.fields:
            if with_docs:
                for line in get_api_field_ref(module, x, field.name):
                    out += [f"    {line}"]
            out += [f"    {get_field(field)},"]
        out += ["]"]

    elif ty is Union:
        out += [
            "class " + get_type_name(x) + "(ctypes.Union): pass",
            f"{get_type_name(x)}._fields_ = [",
        ]
        for variant in x.variants:
            if with_docs:
                for line in get_api_field_ref(module, x, variant.name):
                    out += [f"    {line}"]
            out += [f"    {get_field(variant)},"]
        out += ["]"]

    elif ty is Callback:
        out += [f"{get_type_name(x)} = ctypes.c_void_p"]

    elif ty is Function:
        return_value_type = "None" if x.return_value_type == None else get_type_name(x.return_value_type)
        out += [
            "def " + x.name.snake_case + "("
        ]
        for i, param in enumerate(x.params):
            if i != 0:
                out[-1] += ","
            out += [f"  {get_param(param)}"]
        out += [
            ") -> " + return_value_type + ":",
        ]

        if with_docs:
            for line in get_api_ref(module, x):
                out += [f"    {line}"]
            del out[-1]
            out += [
                "",
                "    Return value: " + return_value_type,
            ]
            if len(x.params) > 0:
                out += [
                    "",
                    "    Parameters:",
                ]
                for param in x.params:
                    out += [
                        f"        {param.name} (`{get_type_name(param.type)}`):"
                    ]
                    for line in get_api_param_ref(module, x, param.name):
                        out += [f"            {line}"]
            out += [
                '    """',
            ]

        out += [
            f"    return _LIB.{x.name.snake_case}(" + ', '.join(str(param.name) for param in x.params) + ")"
        ]

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
        out += ['"""']
        out += [resolve_inline_symbols_to_names(module, x) for x in module.doc.module_doc]
        # Remove the trailing `## API References`.
        del out[-1]
        out += ['"""']

    out += [
        '''import ctypes

def load_taichi_c_api() -> ctypes.CDLL:
    import ctypes.util as ctypes_util
    from os import environ
    from pathlib import Path

    path = ctypes_util.find_library('taichi_c_api')

    if path is None:
        taichi_c_api_install_dir = environ['TAICHI_C_API_INSTALL_DIR']
        if taichi_c_api_install_dir != None:
            candidate_file_names = [
                'bin/taichi_c_api.dll',
                'lib/libtaichi_c_api.so',
                'lib/libtaichi_c_api.dylib',
            ]
            taichi_c_api_install_dir = Path(taichi_c_api_install_dir)
            for candidate_file_name in candidate_file_names:
                candidate_file_path = taichi_c_api_install_dir / candidate_file_name
                if candidate_file_path.exists():
                    path = str(candidate_file_path)
                    break

    if path is None:
        raise RuntimeError(
            'Cannot find taichi_c_api. Please set TAICHI_C_API_INSTALL_DIR environment variable.'
        )

    print(f'Found taichi_c_api at {path}')
    out = ctypes.CDLL(path, ctypes.RTLD_LOCAL)
    return out

_LIB = load_taichi_c_api()
''',
        "",
    ]

    for name, value in module.default_definitions:
        out += [
            f"{name} = {value}",
        ]

    for x in module.declr_reg:
        declr = module.declr_reg.resolve(x)
        out += ["", "", get_declr(module, declr, True)]

    out += [
        "",
    ]

    return "\n".join(out)


def generate_module_header(module):
    if module.is_built_in:
        return

    module_name = module.name.replace("taichi/", "").replace(".h", ".py")
    print(f"processing module '{module_name}'")
    path = f"c_api/python/taichi_runtime/{module_name}"
    with open(path, "w") as f:
        f.write(print_module_header(module))


if __name__ == "__main__":
    builtin_tys = {
        BuiltInType("uint64_t", "ctypes.c_uint64"),
        BuiltInType("int64_t", "ctypes.c_int64"),
        BuiltInType("uint32_t", "ctypes.c_uint32"),
        BuiltInType("int32_t", "ctypes.c_int32"),
        BuiltInType("uint16_t", "ctypes.c_uint16"),
        BuiltInType("int16_t", "ctypes.c_int16"),
        BuiltInType("uint8_t", "ctypes.c_uint8"),
        BuiltInType("int8_t", "ctypes.c_int8"),
        BuiltInType("float", "ctypes.c_float"),
        BuiltInType("const char*", "ctypes.c_void_p"),
        BuiltInType("const char**", "ctypes.c_void_p"),
        BuiltInType("void*", "ctypes.c_void_p"),
        BuiltInType("const void*", "ctypes.c_void_p"),
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


