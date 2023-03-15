from collections import defaultdict
import subprocess
import os
from glob import glob
from typing import Dict, List

# Find C-API objects first.

INSTALL_DIR = "build-taichi-ios-arm64/install/c_api/lib"
OUTPUT_MACHO_PATH = f"{INSTALL_DIR}/libtaichi_c_api.o"
OUTPUT_LIB_PATH = f"{INSTALL_DIR}/libtaichi_c_api.a"

if os.path.exists(OUTPUT_MACHO_PATH):
    os.remove(OUTPUT_MACHO_PATH)

def find_c_api_build_dir() -> str:
    for path, dirs, files in os.walk("build-taichi-ios-arm64/"):
        if "taichi_c_api.build" in path:
            return path
    assert False, "cannot find taichi c-api build directory"

IMPL_DIR = find_c_api_build_dir()
print("c-api object find root is:", IMPL_DIR)

ROOT_OBJS = []
for path, dirs, files in os.walk(IMPL_DIR):
    for file in files:
        if file.endswith(".o"):
            ROOT_OBJS += [f"{path}/{file}"]

print("found the following c-api root objects:")
for x in ROOT_OBJS:
    print(f"  {x}")
print()

# Then enumerate all symbols that might be referenced.

SEARCH_ROOT = "build-taichi-ios-arm64"

class Symbol:
    def __init__(self):
        self.defined_in = []
        self.depended_on_by = []

def dump_symbols(o: str) -> Dict[str, List[str]]:
    symbols = str(subprocess.check_output(["objdump", "--demangle", "--syms", o]), encoding="utf8").splitlines()

    while not symbols[0].startswith("SYMBOL TABLE:"):
        symbols = symbols[1:]
        continue
    symbols = symbols[1:]

    offset = len("0000000000000000         ")
    out = defaultdict(list)
    for x in symbols:
        x = x[offset:]
        k, v = x.split(' ', 1)
        out[k] += [v]
    return out

# symbol name -> obj file paths
SYMBOL_DEFS = defaultdict(list)
# obj file path -> undefined symbol names
UNDEFINED_SYMBOLS = defaultdict(list)
for (base_dir, _, file_names) in os.walk(SEARCH_ROOT, topdown=True):
    # (penguinliong) Ignore shared SPIR-V Tools. We cannot opt out shared
    # library in CMake so we have to do it here.
    if "SPIRV-Tools-shared" in base_dir:
        continue
    for file_name in file_names:
        o = f"{base_dir}/{file_name}"
        if o.endswith(".o"):
            for k, syms in dump_symbols(o).items():
                if k == "*UND*":
                    # Undefined symbols.
                    for x in syms:
                        UNDEFINED_SYMBOLS[o] += [x]
                        SYMBOL_DEFS[x] += []
                else:
                    # Defined symbols.
                    for x in syms:
                        SYMBOL_DEFS[x] += [o]

if False:
    print("all taichi symbols:")
    for symbol_name, sym in SYMBOL_DEFS.items():
        print(f"  [[ {symbol_name} ]]")
        for x in sym:
            print(f"    {x}")

for sym, defs in SYMBOL_DEFS.items():
    defs = set(defs)
    if len(defs) == 0:
        pass
        #print(f"`{sym}` is never defined")
        #assert False
    if len(defs) > 1:
        pass
        #print(f"`{sym}` defined in multiple files: {defs}")
        #assert False
print()

C_API_DEPS = [x for x in ROOT_OBJS]
for o in ROOT_OBJS:
    syms = dump_symbols(o)
    for x in syms["*UND*"]:
        C_API_DEPS += SYMBOL_DEFS[x]

# Cascade dependency and fetch all indirect object dependencies.
LAST_DEPS = str(sorted(C_API_DEPS))
CUR_DEPS = ""
DEPS = sorted(set(C_API_DEPS))
while LAST_DEPS != CUR_DEPS:
    print("cascading...")
    LAST_DEPS = CUR_DEPS

    dep_symbols = []
    for x in DEPS:
        dep_symbols += UNDEFINED_SYMBOLS[x]

    next_deps = DEPS
    for x in set(dep_symbols):
        next_deps += SYMBOL_DEFS[x]

    DEPS = sorted(set(next_deps))
    CUR_DEPS = str(DEPS)
print()

print("dependent objects:")
for dep in set(DEPS):
    print(f"  {dep}")
print()

cmd = ["clang++", "-target", "aarch64-apple-ios13.0", "-r", "-o", OUTPUT_MACHO_PATH] + DEPS
subprocess.check_call(cmd)
print(f"prelinked mach-o to: {OUTPUT_MACHO_PATH}")

if os.path.exists(OUTPUT_LIB_PATH):
    os.remove(OUTPUT_LIB_PATH)
cmd = ["ar", "-crv", OUTPUT_LIB_PATH, OUTPUT_MACHO_PATH]
subprocess.check_call(cmd)
print(f"archive saved to: {OUTPUT_LIB_PATH}")
print()
