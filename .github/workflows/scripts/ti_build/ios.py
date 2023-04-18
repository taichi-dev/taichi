# -*- coding: utf-8 -*-

# -- stdlib --
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

# -- third party --
# -- own --
from . import misc
from .bootstrap import get_cache_home
from .cmake import cmake_args
from .compiler import setup_clang
from .dep import download_dep
from .misc import banner
from .sccache import setup_sccache
from .tinysh import Command, sh


# -- code --
@banner("Setup iOS Build Environment")
def setup_ios(python: Command, pip: Command) -> None:
    s = platform.system()
    if s != "Darwin":
        raise RuntimeError(f"Can only build iOS binaries on macOS, but the current system is {s}.")

    setup_clang()
    setup_sccache()
    pip.install("cmake")

    out = get_cache_home() / "ios-cmake"
    url = "https://raw.githubusercontent.com/leetal/ios-cmake/master/ios.toolchain.cmake"
    download_dep(url, out, force=True, plain=True)

    cmake_args["CMAKE_CONFIGURATION_TYPES"] = "Release"
    cmake_args["CMAKE_TOOLCHAIN_FILE"] = str(out / "ios.toolchain.cmake")
    cmake_args["CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED"] = False
    cmake_args["CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED"] = False
    cmake_args["ENABLE_BITCODE"] = True
    cmake_args["ENABLE_ARC"] = False
    cmake_args["DEPLOYMENT_TARGET"] = "13.0"
    cmake_args["PLATFORM"] = "OS64"
    cmake_args["USE_STDCPP"] = True
    cmake_args["TI_WITH_C_API"] = True
    cmake_args["TI_WITH_METAL"] = True
    cmake_args["TI_WITH_VULKAN"] = False
    cmake_args["TI_WITH_OPENGL"] = False
    cmake_args["TI_WITH_LLVM"] = False
    cmake_args["TI_WITH_CUDA"] = False
    cmake_args["TI_WITH_PYTHON"] = False
    cmake_args["TI_WITH_GGUI"] = False
    cmake_args.writeback()


@banner("Build Taichi iOS C-API Static Library")
def _ios_compile(build_dir: str) -> None:
    """
    Build the Taichi iOS C-API Static Library
    """
    rendered = cmake_args.render()
    defines = [i[1] for i in rendered]
    cmake = sh.cmake
    build_path = Path(build_dir).resolve()
    shutil.rmtree(build_path, ignore_errors=True)
    build_path.mkdir(parents=True, exist_ok=True)

    cmake("-G", "Xcode", "-B", build_path, *defines, ".")
    cmake("--build", build_path, "-t", "taichi_c_api")


@banner("Prelink Taichi iOS C-API Static Library")
def _ios_prelink(build_dir: str, output: str) -> None:
    build_path = Path(build_dir)
    capi_build_path = next(build_path.glob("**/taichi_c_api.build"))
    root_objs = list(capi_build_path.glob("**/*.o"))

    pr = lambda s="": print(s, file=sys.stderr, flush=True)

    misc.info("Found C-API root objects:")
    pr()
    for x in root_objs:
        pr(f"    {x.relative_to(build_path)}")
    pr()

    def dump_symbols(path: str) -> Tuple[List, List]:
        symbols = str(subprocess.check_output(["objdump", "--syms", path]), encoding="utf8").splitlines()
        it = iter(symbols)

        while not next(it).startswith("SYMBOL TABLE:"):
            pass

        defined, undefined = [], []
        for line in it:
            *_, section, sym = line.rsplit()
            col = undefined if section == "*UND*" else defined
            col.append(sym)

        return defined, undefined

    # symbol name -> obj file paths
    SYMBOL_DEFS = {}
    # obj file path -> (defined, undefined)
    OBJECT_SYMBOLS = {}
    for p in build_path.glob("build/**/*.o"):
        sp = str(p.relative_to(build_path))

        # (penguinliong) Ignore shared SPIR-V Tools. We cannot opt out shared
        # library in CMake so we have to do it here.
        if "SPIRV-Tools-shared" in sp:
            continue

        defined, undefined = dump_symbols(str(p))
        OBJECT_SYMBOLS[sp] = (set(defined), set(undefined))

        for sym in defined:
            SYMBOL_DEFS[sym] = sp

    well_known_objs = []
    # well_known_objs.extend(Path('/usr/lib/system').glob('*.dylib'))

    pending_objects = {str(p.relative_to(build_path)) for p in root_objs}
    defined_symbols = set()
    undefined_symbols = set()
    dependencies = set()

    for o in well_known_objs:
        de, _ = dump_symbols(str(o))
        defined_symbols.update(de)

    while True:
        while pending_objects:
            current = pending_objects.pop()
            dependencies.add(current)
            de, und = OBJECT_SYMBOLS[current]
            undefined_symbols.update(und)
            defined_symbols.update(de)

        undefined_symbols -= defined_symbols

        if all(s not in SYMBOL_DEFS for s in undefined_symbols):
            break

        for sym in undefined_symbols:
            if sym in SYMBOL_DEFS:
                pending_objects.add(SYMBOL_DEFS[sym])

    misc.info("Dependent objects:")
    pr()
    for dep in sorted(dependencies):
        pr(f"    {dep}")
    pr()

    if False:
        misc.info("Undefined symbols after prelinking:")
        pr()
        for sym in sorted(undefined_symbols):
            pr(f"    {sym}")
        pr()

    prelinked_obj = build_path / "libtaichi_c_api_prelinked.o"
    prelinked_obj.unlink(missing_ok=True)

    cmd = [
        "clang++",
        "-target",
        "aarch64-apple-ios13.0",
        "-r",
        "-o",
        str(prelinked_obj),
    ]
    cmd.extend(str(build_path / p) for p in dependencies)
    subprocess.check_call(cmd)
    misc.info(f"Prelinked Mach-O to: {prelinked_obj}")

    output_path = Path(output)
    output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ar", "-crv", str(output_path), str(prelinked_obj)]
    subprocess.check_call(cmd)
    misc.info(f"Archive saved to: {output_path}")


def build_ios() -> None:
    with TemporaryDirectory() as tmpdir:
        _ios_compile(tmpdir)
        _ios_prelink(tmpdir, "dist/C-API-iOS/libtaichi_c_api.a")
