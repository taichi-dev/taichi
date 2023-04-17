import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List


class Version:
    def __init__(self, ver: str) -> None:
        if ver.startswith("v"):
            ver = ver[1:]
        xs = [int(x) for x in ver.split(".")]
        assert len(xs) <= 3
        xs += ["0"] * (3 - len(xs))

        self.major = xs[0]
        self.minor = xs[1]
        self.patch = xs[2]

    def to_int(self) -> int:
        version_number = 0
        version_number += self.major
        version_number *= 1000
        version_number += self.minor
        version_number *= 1000
        version_number += self.patch
        return version_number

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class Name:
    def __init__(self, name: str, prefix=[], suffix=[]):
        assert re.match("^[@a-z0-9_]+$", name)
        self._segs = name.split("_")
        self._prefix = prefix
        self._suffix = suffix

    def extend(self, subname) -> "Name":
        if isinstance(subname, str):
            subname = Name(subname)
        assert isinstance(subname, Name)
        assert len(subname._prefix) == 0 and len(subname._suffix) == 0
        return Name("_".join(self._segs + subname._segs), self._prefix, self._suffix)

    @property
    def segs(self):
        return self._prefix + self._segs + self._suffix

    @property
    def snake_case(self) -> str:
        return "_".join(self.segs)

    @property
    def screaming_snake_case(self) -> str:
        return "_".join(x.upper() for x in self.segs)

    @property
    def upper_camel_case(self) -> str:
        return "".join(x.title() for x in self.segs)

    def __repr__(self) -> str:
        return "_".join(self._segs)


class DeclarationRegistry:
    current = None

    def __init__(self, builtin_tys={}):
        # "xxx.yyy" -> Xxx(yyy) Look-up table.
        self._inner = {}
        self._imported = {}
        self._builtin_tys = dict((x.id, x) for x in builtin_tys)

    def resolve(self, id: str) -> "EntryBase":
        if id in self._builtin_tys:
            return self._builtin_tys[id]
        elif id in self._inner:
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


def load_inc_enums():
    paths = glob.glob("taichi/inc/*.inc.h")
    cases = defaultdict(dict)
    for path in paths:
        with open(path) as f:
            for line in f.readlines():
                m = re.match(r"(\w+)\((\w+)\).*", line)
                if m:
                    key = m[1]
                    try:
                        case_name = Name(m[2])
                    except AssertionError:
                        continue
                    cases[key][case_name] = len(cases[key])
    return cases


class EntryBase:
    def __init__(self, j, clazz: str):
        assert "name" in j
        self.vendor = None
        self.is_extension = False
        self.since = None

        prefix = []
        suffix = []
        if "vendor" in j:
            vendor = j["vendor"]
            prefix += ["tix"]
            suffix += [vendor]
            self.vendor = vendor
            self.is_extension = True
        elif "is_extension" in j:
            prefix += ["ti"]
            suffix += ["ext"]
            self.is_extension = True
        else:
            prefix += ["ti"]

        if "version" in j:
            version = int(j["version"])
            if version > 1:
                suffix += [str(version)]
            self.version = version

        if "since" in j:
            self.since = Version(j["since"])

        self.name = Name(j["name"], prefix, suffix)
        self.id = f"{clazz}.{self.name}"


class BuiltInType(EntryBase):
    def __init__(self, id, type_name):
        self.name = "value"
        self.id = id
        self.type_name = type_name


class Alias(EntryBase):
    def __init__(self, j):
        super().__init__(j, "alias")
        self.alias_of = DeclarationRegistry.current.resolve(j["alias_of"])


class Definition(EntryBase):
    def __init__(self, j):
        super().__init__(j, "definition")
        self.value = j["value"]


class Handle(EntryBase):
    def __init__(self, j):
        super().__init__(j, "handle")
        self.is_dispatchable = j["is_dispatchable"]


class Enumeration(EntryBase):
    def __init__(self, j):
        super().__init__(j, "enumeration")
        if "inc_cases" in j:
            self.cases = load_inc_enums()[j["inc_cases"]]
        else:
            self.cases = dict((Name(name), value) for name, value in j["cases"].items())


class BitField(EntryBase):
    def __init__(self, j):
        super().__init__(j, "bit_field")
        if "inc_cases" in j:
            self.bits = load_inc_enums()[j["inc_bits"]]
        else:
            self.bits = dict((Name(name), value) for name, value in j["bits"].items())


class Field:
    def __init__(self, j):
        ty = DeclarationRegistry.current.resolve(j["type"])
        assert ty != None, f"unknown type '{j['type']}'"
        # The type has been registered.
        self.type = ty
        self.name = Name(j["name"]) if "name" in j else ty.name
        self.count = j["count"] if "count" in j else None
        self.by_mut = bool(j["by_mut"]) if "by_mut" in j else False
        self.by_ref = bool(j["by_ref"]) if "by_ref" in j else False


class Structure(EntryBase):
    def __init__(self, j):
        super().__init__(j, "structure")
        self.fields = []
        if "fields" in j:
            for x in j["fields"]:
                self.fields += [Field(x)]


class Union(EntryBase):
    def __init__(self, j):
        super().__init__(j, "union")
        self.variants = []
        if "variants" in j:
            for x in j["variants"]:
                self.variants += [Field(x)]


class Callback(EntryBase):
    def __init__(self, j):
        super().__init__(j, "callback")
        self.return_value_type = None
        self.params = []

        if "parameters" in j:
            for x in j["parameters"]:
                field = Field(x)
                if field.name.snake_case == "@return":
                    self.return_value_type = field.type
                else:
                    self.params += [field]


class Function(EntryBase):
    def __init__(self, j):
        super().__init__(j, "function")
        self.return_value_type = None
        self.params = []
        self.is_device_command = False

        if "parameters" in j:
            for x in j["parameters"]:
                field = Field(x)
                if field.name.snake_case == "@return":
                    self.return_value_type = field.type
                else:
                    self.params += [field]
        if "is_device_command" in j:
            self.is_device_command = True


class Documentation:
    def __init__(self, name: str):
        self.markdown_metadata = []
        self.module_doc = []
        self.api_refs = defaultdict(list)

        path = Path(f"c_api/docs/{name}")
        if path.exists():
            with path.open() as f:
                templ = f.readlines()

            # Ignore markdown headers
            markdown_metadata = []
            if len(templ) > 0 and templ[0].startswith("---"):
                for i in range(1, len(templ)):
                    if templ[i].startswith("---"):
                        i += 1
                        break
                    markdown_metadata += [templ[i].strip()]

            # Skip initial empty lines.
            for i in range(i, len(templ)):
                if len(templ[i].strip()) != 0:
                    break

            # Collect module-level documentation.
            module_doc = []
            for i in range(i, len(templ)):
                line = templ[i].strip()
                module_doc += [line]
                if line.startswith("## API Reference"):
                    break

            # Collect API-references.
            SYM_PATTERN = r"\`(\w+\.\w+)\`"
            FIELD_PATTERN = r"-\s+\`(\w+\.\w+\.\w+)\`:\s*(.*)$"
            cur_sym = None
            api_full_refs = defaultdict(list)
            api_refs = defaultdict(list)
            api_field_refs = defaultdict(str)
            for line in templ[i:]:
                line = line.strip()

                # Match API header
                m = re.match(SYM_PATTERN, line)
                if m:
                    # Remove trailing empty lines.
                    while api_refs[cur_sym] and len(api_refs[cur_sym][-1]) == 0:
                        del api_refs[cur_sym][-1]

                    # Enter parsing for the next symbol.
                    cur_sym = m[1]
                    continue

                api_full_refs[cur_sym] += [line]

                m = re.match(FIELD_PATTERN, line)
                if m:
                    api_field_refs[m[1]] = m[2]
                    continue

                api_refs[cur_sym] += [line]

            self.markdown_metadata = markdown_metadata
            self.module_doc = module_doc
            # Full references including all lines for symbol specification and
            # field specification.
            self.api_full_refs = api_full_refs
            # Symbol specifications without the field specifications.
            self.api_refs = api_refs
            # Field specifications keyed by field symbol triplets.
            self.api_field_refs = api_field_refs


class Module:
    all_modules = {}

    def __init__(self, version: Version, j: dict, builtin_tys: List[BuiltInType]):
        self.name = j["name"]
        self.is_built_in = False
        self.declr_reg = DeclarationRegistry(builtin_tys)
        self.required_modules = []
        self.default_definitions = []
        self.doc = None

        DeclarationRegistry.set_current(self.declr_reg)

        if "default_definitions" in j:
            for jj in j["default_definitions"]:
                name = jj["name"]
                value = jj["value"] if "value" in jj else str(version.to_int())
                self.default_definitions += [(name, value)]

        if "doc" in j:
            self.doc = Documentation(j["doc"])

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
                try:
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
                    elif ty == "callback":
                        self.declr_reg.register(Callback(k))
                    elif ty == "function":
                        self.declr_reg.register(Function(k))
                    else:
                        print(f"ignored unrecognized type declaration '{k}'")
                except:
                    print("failed to generate declaration for:", k)

        DeclarationRegistry.set_current(None)

    @staticmethod
    def load_all(builtin_tys):
        j = None
        with open("c_api/taichi.json") as f:
            j = json.load(f)

        version = Version("v0")
        try:
            with open("version.txt") as f:
                version = Version(f.readline())
        except:
            print("faild to load c-api version")

        print("taichi c-api version is:", version)

        for k in j["modules"]:
            module = Module(version, k, builtin_tys)
            Module.all_modules[module.name] = module

        return list(Module.all_modules.values())
