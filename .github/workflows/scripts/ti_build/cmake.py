# -*- coding: utf-8 -*-

# -- stdlib --
import glob
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Union

# -- third party --
# -- own --
from . import misc
from .escapes import escape_codes
from .misc import banner

# -- code --
OPTION_RE = re.compile(r'option\(([A-Z0-9_]*) +"(.*?)" +(ON|OFF)\)(?: *# wheel-tag: (.*))?')
DEF_RE = re.compile(r"-D([A-Z0-9_]*)(?::BOOL)?=([^ ]+)(?: |$)")


class CMakeArgsManager:
    _VMAP: Dict[Any, Any] = {
        "ON": True,
        "OFF": False,
    }

    def __init__(self, environ_name):
        self.environ_name = environ_name
        self.definitions = {}
        self.option_definitions = {
            "CMAKE_EXPORT_COMPILE_COMMANDS": ("Generate compile_commands.json", False, ""),
        }

        self.finalized = False

    def collect_options(self, *files: str) -> None:
        for fn in files:
            with open(fn, "r") as f:
                for name, desc, default, wheel_tag in OPTION_RE.findall(f.read()):
                    default = self._VMAP.get(default, default)
                    self.option_definitions[name] = (desc, default, wheel_tag)

    def parse_initial_args(self) -> None:
        args = os.environ.get(self.environ_name, "")
        for name, value in DEF_RE.findall(args):
            self.set(name, value)

    def get_effective(self, name: str) -> Union[str, bool]:
        _, default, _ = self.option_definitions.get(name, ("", None, ""))
        return self.definitions.get(name, default)

    def set(self, name: str, value: Union[str, bool]) -> None:
        assert not self.finalized, ".writeback() has been called"
        desc = ""
        value = self._VMAP.get(value, value)
        default = None
        desc, default, wheel_tag = self.option_definitions.get(name, ("", None, ""))
        desc = desc and f" ({desc}) "
        is_bool = isinstance(default, bool)
        assert not is_bool or isinstance(value, bool), f"Option {name} must be bool"

        orig = self.definitions.get(name, default)
        self.definitions[name] = value

        G = escape_codes["bold_green"]
        R = escape_codes["bold_red"]
        B = escape_codes["bold"]
        N = escape_codes["reset"]

        p = lambda s: print(s, file=sys.stderr, flush=True)

        if is_bool:
            if orig != value:
                if value:
                    p(f"{G}:: CMAKE: Enable {name}{desc}{N}")
                else:
                    p(f"{R}:: CMAKE: Disable {name}{desc}{N}")
            else:
                if value:
                    p(f"{B}:: CMAKE: Already enabled: {name}{desc}{N}")
                else:
                    p(f"{B}:: CMAKE: Already disabled: {name}{desc}{N}")
        else:
            assert not wheel_tag, "Set a non boolean value to an option with wheel-tag"
            if orig != value:
                if orig != default:
                    p(f"{R}:: CMAKE- {name}={orig}{desc}{N}")

                p(f"{G}:: CMAKE+ {name}={value}{desc}{N}")

    def render(self) -> List[Tuple[str, str, str]]:
        lst = []
        _map = ("OFF", "ON")
        for name, value in self.definitions.items():
            if isinstance(value, bool):
                v = f"-D{name}:BOOL={_map[value]}"
            else:
                v = f"-D{name}={value}"

            desc, _, _ = self.option_definitions.get(name, ("", None, ""))
            if desc:
                prefix = "DO NOT " if not value else ""
                desc = f" ({prefix}{desc})"

            lst.append((name, v, desc))

        return lst

    def render_wheel_tag(self) -> str:
        tags = []
        for name, (_, default, wheel_tag) in self.option_definitions.items():
            if not wheel_tag:
                continue
            if self.definitions.get(name, default):
                tags.append(wheel_tag)
        return ".".join(sorted(tags))

    @banner("{self.environ_name} Summary")
    def print_summary(self, rendered) -> None:
        p = lambda s: print(s, file=sys.stderr, flush=True)
        misc.info("Effective CMake defines")
        p("")
        G = escape_codes["bold_green"]
        N = escape_codes["reset"]
        for _, v, desc in rendered:
            p(f"    {G}{v}{N}{desc}")
        p("")

    def writeback(self) -> None:
        rendered = self.render()
        self.print_summary(rendered)
        os.environ[self.environ_name] = " ".join([v for _, v, _ in rendered])
        self.finalized = True

    def __setitem__(self, name: str, value: Union[str, bool]) -> None:
        self.set(name, value)

    def __getitem__(self, name: str) -> Union[str, bool]:
        return self.definitions[name]


cmake_args = CMakeArgsManager("TAICHI_CMAKE_ARGS")


@banner("Parsing TAICHI_CMAKE_ARGS")
def _init_cmake_args():
    cmake_args.collect_options("CMakeLists.txt", *glob.glob("cmake/*.cmake"))
    cmake_args.parse_initial_args()


_init_cmake_args()
