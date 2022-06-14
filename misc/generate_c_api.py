import json
import re
from os import system

JSON = None

with open("c_api/taichi.json") as f:
    JSON = json.load(f)

VERSION = JSON["version"]
print("taichi c-api version is:", VERSION)


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


DECLR_REG = None


class Alias:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"alias.{self.name}"
        self.alias_of = j["alias_of"]

    @property
    def type_name(self) -> str:
        return "Ti" + self.name.upper_camel_case

    def declr(self):
        return f"typedef {self.alias_of} {self.type_name};"


class Definition:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"definition.{self.name}"
        self.value = j["value"]

    def declr(self):
        return f"#define {self.name.screaming_snake_case} {self.value}"


class Handle:
    def __init__(self, j):
        self.name = Name(j["name"])
        self.id = f"handle.{self.name}"
        self.is_dispatchable = j["is_dispatchable"]

    @property
    def type_name(self) -> str:
        return "Ti" + self.name.upper_camel_case

    def declr(self):
        return f"typedef struct {self.type_name}_t* {self.type_name};"


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
        out = ["typedef enum " + self.type_name + " {"]
        for name, value in self.cases.items():
            out += [f"  {name} = {value},"]
        out += [f"  {self.get_case_name(Name('max_enum'))} = 0xffffffff,"]
        out += ["} " + self.type_name + ";"]
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
        self.bits = bits

    @property
    def type_name(self):
        return "Ti" + self.name.upper_camel_case + "FlagBits"

    @property
    def field_type_name(self):
        return "Ti" + self.name.upper_camel_case + "Flags"

    def get_flag_name(self, flag_name: Name):
        return "TI_" + self.name.screaming_snake_case + "_" + flag_name.screaming_snake_case + "_BIT"

    def declr(self):
        out = ["typedef enum " + self.type_name + " {"]
        for name, value in self.bits.items():
            out += [f"  {name} = {value},"]
        out += ["} " + self.type_name + ";"]
        out += [f"typedef TiFlags {self.field_type_name};"]
        return '\n'.join(out)


class CType:
    def __init__(self, name: str):
        self.name = name

    @property
    def type_name(self):
        return self.name


class Field:
    def __init__(self, j):
        ty = DECLR_REG.resolve(j["type"])
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

    def declr(self):
        # `count` is an integer so it's a static array.
        is_dyn_array = self.count and not isinstance(self.count, int)

        is_ptr = self.by_ref or self.by_mut or is_dyn_array
        const_q = "const" if not self.by_mut else ""

        if is_ptr:
            return f"{const_q} {self.type.type_name}* {self.name}"
        elif self.count:
            return f"{self.type.type_name} {self.name}[{self.count}]"
        else:
            return f"{self.type.type_name} {self.name}"


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
        out = ["typedef struct " + self.type_name + " {"]
        for x in self.fields:
            out += [f"  {x.declr()};"]
        out += ["} " + self.type_name + ";"]
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
        out = ["typedef union " + self.type_name + " {"]
        for x in self.variants:
            out += [f"  {x.declr()};"]
        out += ["} " + self.type_name + ";"]
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
    def func_name(self):
        name = "ti_" + self.name.snake_case
        if self.is_extension:
            name += "_ext"
        if self.version > 1:
            name += f"_{self.version}"
        return name

    def declr(self):
        return_value_type = "void" if self.return_value_type == None else self.return_value_type.type_name
        out = [
            "TI_DLL_EXPORT " + return_value_type + " TI_API_CALL " +
            self.func_name + "("
        ]
        out += [',\n'.join(f"  {param.declr()}" for param in self.params)]
        out += [");"]
        return '\n'.join(out)


MODULES = {}


class Module:
    def __init__(self, j):
        self.is_built_in = False
        self.declr_reg = DeclarationRegistry()
        self.required_modules = []

        global DECLR_REG
        DECLR_REG = self.declr_reg

        if "is_built_in" in j:
            self.is_built_in = True
            # Built-in headers are hand-written so we can return right away.
            return

        if "required_modules" in j:
            for x in j["required_modules"]:
                assert x in MODULES
                module = MODULES[x]
                self.declr_reg.import_declrs(module.declr_reg)
                self.required_modules += [x]

        if "declarations" in j:
            for k in j["declarations"]:
                ty = k["type"]

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

        DECLR_REG = None

    def declr(self):
        out = ["#pragma once"]

        for x in self.required_modules:
            out += [f"#include <{x}>"]

        out += [
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif // __cplusplus",
            "",
        ]

        for x in self.declr_reg:
            out += [
                "",
                f"// {x}",
                self.declr_reg.resolve(x).declr(),
            ]

        out += [
            "",
            "#ifdef __cplusplus",
            '} // extern "C"',
            "#endif // __cplusplus",
            "",
        ]

        return '\n'.join(out)


def generate_module_header(j):
    module_name = j["name"]
    module = Module(j)
    MODULES[module_name] = module

    if module.is_built_in:
        return

    print(f"processing module '{module_name}'")
    path = f"c_api/include/{module_name}"
    with open(path, "w") as f:
        f.write(module.declr())

    system(f"clang-format {path} -i")


for module in JSON["modules"]:
    generate_module_header(module)
