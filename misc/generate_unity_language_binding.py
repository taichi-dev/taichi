import json
import re

#from os import system


TYPE_MAP = {
    "void": "void",
    "int32_t": "int",
    "uint32_t": "uint",
    "int64_t": "long",
    "uint64_t": "ulong",
    "float": "float",
    "const char*": "string",
    "void*": "IntPtr",
}

class InternalAlias:
    def __init__(self, name: str):
        self.name = name
    @property
    def type_name(self):
        self.name


class Name:
    def __init__(self, name: str):
        assert re.match('^[@a-z0-9_]+$', name)
        self._segs = name.split("_")

    @property
    def snake_case(self) -> str:
        return '_'.join(self._segs)

    @property
    def screaming_snake_case(self) -> str:
        return '_'.join(x.upper() for x in self._segs)

    @property
    def upper_camel_case(self) -> str:
        return ''.join(x.title() for x in self._segs)

    def __repr__(self) -> str:
        return self.snake_case


class DeclarationRegistry:
    current = None

    def __init__(self):
        # "xxx.yyy" -> Xxx(yyy) Look-up table.
        self._inner = {}
        self._imported = {}

    def resolve(self, id: str):
        if id in self._inner:
            return self._inner[id]
        elif id in self._imported:
            return self._imported[id]
        else:
            return None

    def register(self, x):
        self._inner[x.id] = x

    def import_declrs(self, other):
        for x in other._inner.values():
            self._imported[x.id] = x

    def __iter__(self):
        return iter(self._inner)

    @staticmethod
    def set_current(declr_reg):
        DeclarationRegistry.current = declr_reg


class Alias:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"alias.{self.name}"
        self.alias_of = j["alias_of"]
        TYPE_MAP[self.type_name] = TYPE_MAP[self.alias_of]

    @property
    def type_name(self) -> str:
        return TYPE_MAP[self.alias_of]

    def declr(self):
        return f"// using Ti{self.name.upper_camel_case} = {TYPE_MAP[self.alias_of]};"


class Definition:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"definition.{self.name}"
        self.value = j["value"]

    def declr(self):
        out = [
            "static partial class Def {",
            f"public const uint {self.name.screaming_snake_case} = {self.value};",
            "}"
        ]
        return '\n'.join(out)


class Handle:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"handle.{self.name}"
        self.is_dispatchable = j["is_dispatchable"]

    @property
    def type_name(self) -> str:
        return "Ti" + self.name.upper_camel_case

    def declr(self):
        out = [
            "[StructLayout(LayoutKind.Sequential)]",
            "public struct " + self.type_name + " {",
            "  public IntPtr Inner;",
            "}",
        ]
        return '\n'.join(out)


class Enumeration:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"enumeration.{self.name}"

        cases = {}
        if "inc_cases" in j:
            inc_file_name = "taichi/inc/" + j["inc_cases"] + ".inc.h"
            with open(inc_file_name) as f:
                for line in f.readlines():
                    m = re.match(r"\w+\((\w+)\).*", line)
                    if m:
                        case_name = self.get_case_name(Name(m[1]))
                        cases[case_name] = len(cases)
        else:
            for name, value in j["cases"].items():
                cases[self.get_case_name(Name(name))] = value
        self.cases = cases

    @property
    def type_name(self):
        return "Ti" + self.name.upper_camel_case

    def get_case_name(self, case_name: Name):
        return "TI_" + self.name.screaming_snake_case + "_" + case_name.screaming_snake_case

    def declr(self):
        out = ["public enum " + self.type_name + " {"]
        for name, value in self.cases.items():
            out += [f"  {name} = {value},"]
        out += [f"  {self.get_case_name(Name('max_enum'))} = 0x7fffffff,"]
        out += ["}"]
        return '\n'.join(out)


class BitField:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"bit_field.{self.name}"

        bits = {}
        if "inc_bits" in j:
            inc_file_name = "taichi/inc/" + j["inc_bits"] + ".inc.h"
            with open(inc_file_name) as f:
                for line in enumerate(f.readlines()):
                    m = re.match(r"\w+\((\w+)\).*", line)
                    if m:
                        bit_name = self.get_flag_name(Name(m[1]))
                        bits[bit_name] = len(bits)
        else:
            for name, value in j["bits"].items():
                bits[self.get_flag_name(Name(name))] = value
        TYPE_MAP[self.type_name] = "uint"
        self.bits = bits

    @property
    def type_name(self):
        return "Ti" + self.name.upper_camel_case + "FlagBits"

    @property
    def field_type_name(self):
        return self.name.upper_camel_case + "Flags"

    def get_flag_name(self, flag_name: Name):
        return "TI_" + self.name.screaming_snake_case + "_" + flag_name.screaming_snake_case + "_BIT"

    def declr(self):
        out = [
            "[Flags]",
            "public enum " + self.type_name + " {"
        ]
        for name, value in self.bits.items():
            out += [f"  {name} = 1 << {value},"]
        out += ["};"]
        return '\n'.join(out)


class CType:
    def __init__(self, name: str):
        self.name = TYPE_MAP[name]

    @property
    def type_name(self):
        return self.name


class Field:
    def __init__(self, j):
        ty = DeclarationRegistry.current.resolve(j["type"])
        if ty != None:
            # The type has been registered.
            self.type = ty
            if "name" in j:
                self.name = Name(j["name"])
            else:
                self.name = ty.name
        else:
            # The type is not (yet) registered, treat it as a untracked C type.
            self.type = CType(j["type"])
            self.name = Name(j["name"])
        self.count = j["count"] if "count" in j else None
        self.by_mut = j["by_mut"] if "by_mut" in j else None
        self.by_ref = j["by_ref"] if "by_ref" in j else None

    def declr_c_function_param(self):
        # `count` is an integer so it's a static array.
        is_dyn_array = self.count and not isinstance(self.count, int)

        if self.by_ref or self.by_mut:
            return f"IntPtr {self.name}"
        elif is_dyn_array:
            return f"[MarshalAs(UnmanagedType.LPArray)] {self.type.type_name}[] {self.name}"
        elif self.count:
            return f"{self.type.type_name}[{self.count}] {self.name}"
        else:
            return f"{self.type.type_name} {self.name}"

    def declr_function_param(self):
        # `count` is an integer so it's a static array.
        is_dyn_array = self.count and not isinstance(self.count, int)

        is_ptr = self.by_ref or self.by_mut or is_dyn_array

        if is_dyn_array:
            return f"{self.type.type_name}[] {self.name}"
        elif self.count:
            return f"{self.type.type_name}[{self.count}] {self.name}"
        else:
            return f"{self.type.type_name} {self.name}"

    def declr_struct_field(self):
        # `count` is an integer so it's a static array.
        is_dyn_array = self.count and not isinstance(self.count, int)

        is_ptr = self.by_ref or self.by_mut or is_dyn_array

        out = ""
        if is_ptr:
            out += f"IntPtr {self.name}"
        elif self.count:
            out += f"[MarshalAs(UnmanagedType.ByValArray, SizeConst={self.count})] "
            out += f"public {self.type.type_name}[] {self.name}"
        else:
            out += f"public {self.type.type_name} {self.name}"
        return out

    def declr_union_variant(self):
        # `count` is an integer so it's a static array.
        is_dyn_array = self.count and not isinstance(self.count, int)

        is_ptr = self.by_ref or self.by_mut or is_dyn_array

        out = "[FieldOffset(0)] "
        if is_ptr:
            out += f"IntPtr {self.name}"
        elif self.count:
            out += f"[MarshalAs(UnmanagedType.ByValArray, SizeConst={self.count})] "
            out += f"public {self.type.type_name}[] {self.name}"
        else:
            out += f"public {self.type.type_name} {self.name}"
        return out


class Structure:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"structure.{self.name}"
        self.fields = []
        if "fields" in j:
            for x in j["fields"]:
                self.fields += [Field(x)]

    @property
    def type_name(self):
        return "Ti" + self.name.upper_camel_case

    def declr(self):
        out = [
            "[StructLayout(LayoutKind.Sequential)]",
            "public struct " + self.type_name + " {",
        ]
        for x in self.fields:
            out += [f"  {x.declr_struct_field()};"]
        out += ["}"]
        return '\n'.join(out)


class Union:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"union.{self.name}"
        self.variants = []
        if "variants" in j:
            for x in j["variants"]:
                self.variants += [Field(x)]

    @property
    def type_name(self):
        return "Ti" + self.name.upper_camel_case

    def declr(self):
        out = [
            "[StructLayout(LayoutKind.Explicit)]",
            "public struct " + self.type_name + " {",
        ]
        for x in self.variants:
            out += [f"  {x.declr_union_variant()};"]
        out += ["}"]
        return '\n'.join(out)


class Function:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"function.{self.name}"
        self.version = 1
        self.is_extension = False
        self.return_value_type = None
        self.params = []

        if "version" in j:
            self.version = j["version"]

        if "is_extension" in j:
            self.is_extension = j["is_extension"]

        if "parameters" in j:
            for x in j["parameters"]:
                field = Field(x)
                if field.name.snake_case == "@return":
                    self.return_value_type = field.type
                else:
                    self.params += [field]

    @property
    def c_func_name(self):
        name = "ti_" + self.name.snake_case
        if self.is_extension:
            name += "_ext"
        if self.version > 1:
            name += f"_{self.version}"
        return name
    @property
    def func_name(self):
        name = self.name.upper_camel_case
        if self.is_extension:
            name += "_ext"
        if self.version > 1:
            name += f"_{self.version}"
        return name

    def declr(self):
        return_value_type = "void" if self.return_value_type == None else self.return_value_type.type_name
        out = [
            "static partial class Ffi {",
            "#if (UNITY_IOS || UNITY_TVOS || UNITY_WEBGL) && !UNITY_EDITOR",
            '    [DllImport ("__Internal")]',
            "#else",
            '    [DllImport("taichi_c_api")]',
            "#endif",
            "private static extern " + return_value_type + " " + self.c_func_name + "(",
            ',\n'.join(f"  {param.declr_c_function_param()}" for param in self.params),
            ");",
            "public static " + return_value_type + " " + self.func_name + "(",
            ',\n'.join(f"  {param.declr_function_param()}" for param in self.params),
            ") {",
        ]
        for param in self.params:
            if (isinstance(param.type, Structure) or isinstance(param.type, Union)) and not param.count:
                out += [
                    f"  IntPtr hglobal_{param.name} = Marshal.AllocHGlobal(Marshal.SizeOf(typeof({param.type.type_name})));",
                    f"  Marshal.StructureToPtr({param.name}, hglobal_{param.name}, false);"
                ]
        if self.return_value_type:
            out += [f"  var rv = {self.c_func_name}("]
        else:
            out += [f"  {self.c_func_name}("]
        for i, param in enumerate(self.params):
            if (isinstance(param.type, Structure) or isinstance(param.type, Union)) and not param.count:
                out += [f"    hglobal_{param.name}{','if i + 1 != len(self.params) else ''}"]
            else:
                out += [f"    {param.name}{','if i + 1 != len(self.params) else ''}"]
        out += ["  );"]
        for param in self.params:
            if (isinstance(param.type, Structure) or isinstance(param.type, Union)) and not param.count:
                out += [
                    f"  Marshal.FreeHGlobal(hglobal_{param.name});"
                ]
        if self.return_value_type:
            out += [f"  return rv;"]
        out += [
            "}",
            "}"
        ]
        return '\n'.join(out)


class Module:
    current = None
    all_modules = {}

    def __init__(self, j):
        Module.current = self
        self.name = Name(j["name"][len("taichi/"):-len(".h")])
        self.is_built_in = False
        self.declr_reg = DeclarationRegistry()
        self.required_modules = []

        DeclarationRegistry.set_current(self.declr_reg)

        if "is_built_in" in j:
            self.is_built_in = True
            # Built-in headers are hand-written so we can return right away.
            return

        if "required_modules" in j:
            for x in j["required_modules"]:
                assert x in Module.all_modules
                module = Module.all_modules[x]
                self.declr_reg.import_declrs(module.declr_reg)
                self.required_modules += [x]

        if "declarations" in j:
            for k in j["declarations"]:
                ty = k["type"]

                try:
                    if ty == "alias":
                        self.declr_reg.register(Alias(k))
                    elif ty == "definition":
                        self.declr_reg.register(Definition(k))
                    elif ty == "handle":
                        self.declr_reg.register(Handle(k))
                    elif ty == "enumeration":
                        self.declr_reg.register(Enumeration(k))
                    elif ty == "bit_field":
                        self.declr_reg.register(BitField(k))
                    elif ty == "structure":
                        self.declr_reg.register(Structure(k))
                    elif ty == "union":
                        self.declr_reg.register(Union(k))
                    elif ty == "function":
                        self.declr_reg.register(Function(k))
                    else:
                        print(f"ignored unrecognized type declaration '{k}'")
                except KeyError as k:
                    print(f"Ignored declaration '{x}' with hidden type: {k}")

        DeclarationRegistry.set_current(None)

    def declr(self):
        out = [
            "using System;",
            "using System.Runtime.InteropServices;",
            "using System.Collections.Generic;",
            "",
            "namespace Taichi {",
        ]

        for x in self.declr_reg:
                out += [
                    "",
                    f"// {x}",
                    self.declr_reg.resolve(x).declr(),
                ]

        out += [
            "",
            "} // namespace Taichi",
            "",
        ]

        return '\n'.join(out)

    @staticmethod
    def generate_header(j):
        module_name = j["name"]
        module = Module(j)
        Module.all_modules[module_name] = module

        if module.is_built_in:
            return

        print(f"processing module '{module_name}'")
        assert re.match("taichi/\w+.h", module_name)
        module_name = module_name[len("taichi/"):-len(".h")]
        path = f"c_api/unity/{module_name}.cs"
        with open(path, "w") as f:
            f.write(module.declr())

        #system(f"clang-format {path} -i")


if __name__ == "__main__":
    j = None
    with open("c_api/taichi.json") as f:
        j = json.load(f)

    version = j["version"]
    print("taichi c-api version is:", version)

    for module in j["modules"]:
        Module.generate_header(module)
