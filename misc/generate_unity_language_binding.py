from taichi_json import EntryBase, BuiltInType, Alias, Handle, Definition, \
    Handle, Enumeration, BitField, Field, Structure, Union, Function, Module
import re


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


def get_c_function_param(x: Field):
    # `count` is an integer so it's a static array.
    is_dyn_array = x.count and not isinstance(x.count, int)

    if x.by_ref or x.by_mut:
        return f"IntPtr {x.name}"
    elif is_dyn_array:
        return f"[MarshalAs(UnmanagedType.LPArray)] {get_type_name(x.type)}[] {x.name}"
    elif x.count:
        return f"{get_type_name(x.type)}[{x.count}] {x.name}"
    else:
        return f"{get_type_name(x.type)} {x.name}"


def get_function_param(x: Field):
    # `count` is an integer so it's a static array.
    is_dyn_array = x.count and not isinstance(x.count, int)

    if is_dyn_array:
        return f"{get_type_name(x.type)}[] {x.name}"
    elif x.count:
        return f"{get_type_name(x.type)}[{x.count}] {x.name}"
    else:
        return f"{get_type_name(x.type)} {x.name}"


def get_struct_field(x: Field):
    # `count` is an integer so it's a static array.
    is_dyn_array = x.count and not isinstance(x.count, int)

    is_ptr = x.by_ref or x.by_mut or is_dyn_array

    out = ""
    if is_ptr:
        out += f"IntPtr {x.name}"
    elif x.count:
        out += f"[MarshalAs(UnmanagedType.ByValArray, SizeConst={x.count})] "
        out += f"public {get_type_name(x.type)}[] {x.name}"
    else:
        out += f"public {get_type_name(x.type)} {x.name}"
    return out


def get_union_variant(x: Field):
    # `count` is an integer so it's a static array.
    is_dyn_array = x.count and not isinstance(x.count, int)

    is_ptr = x.by_ref or x.by_mut or is_dyn_array

    out = "[FieldOffset(0)] "
    if is_ptr:
        out += f"IntPtr {x.name}"
    elif x.count:
        out += f"[MarshalAs(UnmanagedType.ByValArray, SizeConst={x.count})] "
        out += f"public {get_type_name(x.type)}[] {x.name}"
    else:
        out += f"public {get_type_name(x.type)} {x.name}"
    return out


def get_declr(x: EntryBase):
    ty = type(x)
    if ty is BuiltInType:
        return ""

    elif ty is Alias:
        return f"// using {get_type_name(x)} = {get_type_name(x.alias_of)};"

    elif ty is Definition:
        out = [
            "static partial class Def {",
            f"public const uint {x.name.screaming_snake_case} = {x.value};",
            "}"
        ]
        return '\n'.join(out)

    elif ty is Handle:
        out = [
            "[StructLayout(LayoutKind.Sequential)]",
            "public struct " + get_type_name(x) + " {",
            "  public IntPtr Inner;",
            "}",
        ]
        return '\n'.join(out)

    elif ty is Enumeration:
        out = ["public enum " + get_type_name(x) + " {"]
        for name, value in x.cases.items():
            out += [f"  {name.screaming_snake_case} = {value},"]
        out += [f"  {x.name.extend('max_enum').screaming_snake_case} = 0x7fffffff,"]
        out += ["}"]
        return '\n'.join(out)

    elif ty is BitField:
        out = ["[Flags]", "public enum " + get_type_name(x) + " {"]
        for name, value in x.bits.items():
            out += [f"  {name.extend('bit').screaming_snake_case} = 1 << {value},"]
        out += ["};"]
        return '\n'.join(out)

    elif ty is Structure:
        out = [
            "[StructLayout(LayoutKind.Sequential)]",
            "public struct " + get_type_name(x) + " {",
        ]
        for field in x.fields:
            out += [f"  {get_struct_field(field)};"]
        out += ["}"]
        return '\n'.join(out)

    elif ty is Union:
        out = [
            "[StructLayout(LayoutKind.Explicit)]",
            "public struct " + get_type_name(x) + " {",
        ]
        for variant in x.variants:
            out += [f"  {get_union_variant(variant)};"]
        out += ["}"]
        return '\n'.join(out)

    elif ty is Function:
        return_value_type = "void" if x.return_value_type == None else get_type_name(
            x.return_value_type)
        out = [
            "static partial class Ffi {",
            "#if (UNITY_IOS || UNITY_TVOS || UNITY_WEBGL) && !UNITY_EDITOR",
            '    [DllImport ("__Internal")]',
            "#else",
            '    [DllImport("taichi_unity")]' if x.vendor == "unity" else '    [DllImport("taichi_c_api")]',
            "#endif",
            "private static extern " + return_value_type + " " +
            x.name.snake_case + "(",
            ',\n'.join(f"  {get_c_function_param(param)}"
                       for param in x.params),
            ");",
            "public static " + return_value_type +
            " " + x.name.upper_camel_case + "(",
            ',\n'.join(f"  {get_function_param(param)}"
                       for param in x.params),
            ") {",
        ]
        for param in x.params:
            if (isinstance(param.type, Structure)
                    or isinstance(param.type, Union)) and not param.count:
                out += [
                    f"  IntPtr hglobal_{param.name} = Marshal.AllocHGlobal(Marshal.SizeOf(typeof({get_type_name(param.type)})));",
                    f"  Marshal.StructureToPtr({param.name}, hglobal_{param.name}, false);"
                ]
        if x.return_value_type:
            out += [f"  var rv = {x.name.snake_case}("]
        else:
            out += [f"  {x.name.snake_case}("]
        for i, param in enumerate(x.params):
            if (isinstance(param.type, Structure)
                    or isinstance(param.type, Union)) and not param.count:
                out += [
                    f"    hglobal_{param.name}{','if i + 1 != len(x.params) else ''}"
                ]
            else:
                out += [
                    f"    {param.name}{','if i + 1 != len(x.params) else ''}"
                ]
        out += ["  );"]
        for param in x.params:
            if (isinstance(param.type, Structure)
                    or isinstance(param.type, Union)) and not param.count:
                out += [f"  Marshal.FreeHGlobal(hglobal_{param.name});"]
        if x.return_value_type:
            out += [f"  return rv;"]
        out += ["}", "}"]
        return '\n'.join(out)

    else:
        raise RuntimeError(f"'{x.id}' doesn't need declaration")


def print_module_header(module):
    out = [
        "using System;",
        "using System.Runtime.InteropServices;",
        "using System.Collections.Generic;",
        "",
        "namespace Taichi {",
    ]

    for x in module.declr_reg:
        out += [
            "",
            f"// {x}",
            get_declr(module.declr_reg.resolve(x)),
        ]

    out += [
        "",
        "} // namespace Taichi",
        "",
    ]

    return '\n'.join(out)


def generate_module_header(module):
    if module.is_built_in:
        return

    print(f"processing module '{module.name}'")
    assert re.match("taichi/\w+.h", module.name)
    module_name = module.name[len("taichi/"):-len(".h")]
    path = f"c_api/unity/{module_name}.cs"
    with open(path, "w") as f:
        f.write(print_module_header(module))


if __name__ == "__main__":
    builtin_tys = {
        BuiltInType("void", "void"),
        BuiltInType("int32_t", "int"),
        BuiltInType("uint32_t", "uint"),
        BuiltInType("int64_t", "long"),
        BuiltInType("uint64_t", "ulong"),
        BuiltInType("float", "float"),
        BuiltInType("const char*", "string"),
        BuiltInType("void*", "IntPtr"),
        BuiltInType("const void*", "IntPtr"),
        BuiltInType("alias.bool", "uint"),
    }

    for module in Module.load_all(builtin_tys):
        generate_module_header(module)
