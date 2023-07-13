import os, sys

from typing import Mapping, List, Union
from pycparser import parse_file
from pycparser.c_ast import Decl as CASTDecl, NodeVisitor


CCORE_PYTHON_FILE_FORMAT = """import os
import sys
import ctypes

from taichi._lib.utils import package_root

# EXPORTED_FUNCTIONS = (...)
{exported_functions}


def _load_dll(path):
    try:
        if (
            sys.version_info[0] > 3
            or sys.version_info[0] == 3
            and sys.version_info[1] >= 8
        ):
            dll = ctypes.CDLL(path, winmode=0)
        else:
            dll = ctypes.CDLL(path)
    except OSError:
        return None
    return dll


def load_core_exports_dll():
    bin_path = os.path.join(package_root, "_lib", "core_exports", "bin")
    if os.name == "nt":
        if (
            sys.version_info[0] > 3
            or sys.version_info[0] == 3
            and sys.version_info[1] >= 8
        ):
            os.add_dll_directory(bin_path)
        else:
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        dll_path = os.path.join(bin_path, "taichi_core_exports.dll")
    elif sys.platform == "darwin":
        dll_path = os.path.join(bin_path, "libtaichi_core_exports.dylib")
    else:
        dll_path = os.path.join(bin_path, "taichi_core_exports.so")

    return _load_dll(dll_path)


class TaichiCCore:
    def __init__(self) -> None:
        self._dll = load_core_exports_dll()
        if self._dll is None:
            raise RuntimeError("Cannot load taichi_core_exports.dll")

        global EXPORTED_FUNCTIONS
        for func in EXPORTED_FUNCTIONS:
            func_name, func_argtypes, func_restype = func
            c_func = getattr(self._dll, func_name)
            c_func.argtypes = func_argtypes
            c_func.restype = func_restype
        del EXPORTED_FUNCTIONS

    def __getattr__(self, name):
        return getattr(self._dll, name)


taichi_ccore = TaichiCCore()

__all__ = ['taichi_ccore']
"""


C_BUILTIN_TYPE_TO_CTYPES_TYPE = {
    "int": "ctypes.c_int",
    "float": "ctypes.c_float",
    "double": "ctypes.c_double",
    "int64_t": "ctypes.c_int64",
    "uint64_t": "ctypes.c_uint64",
    "size_t": "ctypes.c_size_t",
    "uintptr_t": "ctypes.c_uint64",
}


Type = CASTDecl


def is_cstr_type(type: Type):
    from pycparser.c_ast import PtrDecl, TypeDecl, IdentifierType

    return (
        isinstance(type.type, PtrDecl)
        and isinstance(type.type.type, TypeDecl)
        and isinstance(type.type.type.type, IdentifierType)
        and type.type.type.type.names[0] == "char"
    )


class EnumDecl:
    def __init__(self, name: str, values: Mapping[str, int]):
        self.name = name
        self.values = values


class FuncParameter:
    def __init__(self, name: str, type: Type):
        self.name = name
        self.type = type

    def is_in_param(self):
        return not self.is_out_param()

    def is_out_param(self):
        return self.name.startswith("ret_")

    def is_arr_param(self):
        return self.name.startswith("ap_")

    def is_cstr_param(self):
        return is_cstr_type(self.type)


class FuncDecl:
    def __init__(
        self,
        name: str,
        in_params: List[FuncParameter],
        out_params: List[FuncParameter],
        ret_type: Type,
    ):
        self.name = name
        self.in_params = in_params
        self.out_params = out_params
        self.ret_type = ret_type


class ClassMethodDecl:
    CONSTRACTOR_TYPE = "constructor"
    DESTRUCTOR_TYPE = "destructor"
    METHOD_TYPE = "method"
    STATIC_METHOD_TYPE = "staticmethod"

    def __init__(
        self,
        classname: str,
        method_name: str,
        original_func_decl: FuncDecl,
        type: str = None,
    ):
        self.classname = classname
        self.method_name = method_name
        self.original_func_decl = original_func_decl
        self.type = type or ClassMethodDecl.METHOD_TYPE


class ClassDecl:
    def __init__(self, name: str, methods: Mapping[str, ClassMethodDecl]):
        self.name = name
        self.methods = methods


class ExportsHeader:  # exports.h
    def __init__(self, enums: List[EnumDecl], funcs: List[FuncDecl]):
        self.enums = enums
        self.funcs = funcs


class ConstantEvalVisitor(NodeVisitor):
    def __init__(self):
        self.value = None

    def visit_UnaryOp(self, node):
        if node.op == "-":
            self.visit(node.expr)
            self.value = f"-{self.value}"

    def visit_Constant(self, node):
        self.value = node.value


class ExportsHeaderASTVisitor(NodeVisitor):
    def __init__(self):
        self.enums = []
        self.funcs = []

    def visit_Decl(self, node):
        from pycparser import c_ast

        assert isinstance(node, CASTDecl)
        if not isinstance(node.type, c_ast.FuncDecl):
            self.generic_visit(node)
            return

        func_name = node.name
        c_ast_func_decl = node.type
        args = c_ast_func_decl.args
        ret_type = c_ast_func_decl.type
        in_params, out_params = [], []
        for p in args.params:
            param = FuncParameter(name=p.name, type=p)
            if param.is_in_param():
                in_params.append(param)
            elif param.is_out_param():
                out_params.append(param)
            else:
                raise ValueError(f"Unknown type: {param.type}")
        func_decl = FuncDecl(
            name=func_name,
            in_params=in_params,
            out_params=out_params,
            ret_type=ret_type,
        )
        self.funcs.append(func_decl)

    def visit_Enum(self, node):
        from pycparser import c_ast

        assert isinstance(node, c_ast.Enum)
        enum_name = node.name
        if enum_name.startswith("Tie"):
            values = {}
            for enumerator in node.values.enumerators:
                assert isinstance(enumerator, c_ast.Enumerator)
                visitor = ConstantEvalVisitor()
                visitor.visit(enumerator.value)
                values[enumerator.name] = visitor.value
            enum_decl = EnumDecl(name=enum_name, values=values)
            self.enums.append(enum_decl)


def parse_exports_header(filename: str) -> ExportsHeader:
    assert filename.endswith(".h")
    # FIXME: DON'T hardcode the path and args here!
    ast = parse_file(
        filename,
        use_cpp=True,
        cpp_path="D:/programming_tools_/LLVM14/bin/clang.exe",
        cpp_args=[
            "-DTI_EXPORTS_TO_PY",
            "-E",
            "-ID:/programming_tools_/pycparser-master/utils/fake_libc_include",
        ],
    )

    visitor = ExportsHeaderASTVisitor()
    visitor.visit(ast)
    return ExportsHeader(enums=visitor.enums, funcs=visitor.funcs)


def translate_c_type_to_ctypes_type(type: Type, exclude_ptr_levels: int = 0) -> str:
    from pycparser.c_ast import TypeDecl, IdentifierType, PtrDecl

    ctype = type.type if isinstance(type, Type) else type
    if isinstance(ctype, TypeDecl):
        return translate_c_type_to_ctypes_type(ctype.type, exclude_ptr_levels)
    elif isinstance(ctype, IdentifierType):
        typename = ctype.names[0]
        if typename in C_BUILTIN_TYPE_TO_CTYPES_TYPE:
            return C_BUILTIN_TYPE_TO_CTYPES_TYPE[typename]
        if typename.startswith("Tie") and typename.endswith("Handle"):
            return "ctypes.c_void_p"
    elif isinstance(ctype, PtrDecl):
        if exclude_ptr_levels > 0:
            return translate_c_type_to_ctypes_type(ctype.type, exclude_ptr_levels - 1)
        if isinstance(ctype.type.type, IdentifierType):
            typename = ctype.type.type.names[0]
            if typename == "char":
                return "ctypes.c_char_p"
        return f"ctypes.POINTER({translate_c_type_to_ctypes_type(ctype.type, 0)})"
    else:
        raise ValueError(f"Unknown type: {ctype}")


def translate_func_decl(
    func: FuncDecl,
) -> Union[ClassMethodDecl, FuncDecl]:
    funcname = func.name
    splited_by_ = funcname.split("_")
    if len(splited_by_) < 3:
        raise ValueError(
            f"Invliad function name: {funcname}, which has less than 3 parts separated by '_'. (e.g. tie_Kernel_create)"
        )
    tie = splited_by_[0]
    if tie != "tie":
        raise ValueError(
            f"Invliad function name: {funcname}, which is not started with 'tie'"
        )
    class_name = splited_by_[1]
    method_name = "_".join(splited_by_[2:])

    if class_name == "G":
        return func  # Global function

    if method_name == "create":
        return ClassMethodDecl(
            classname=class_name,
            method_name=method_name,
            original_func_decl=func,
            type=ClassMethodDecl.CONSTRACTOR_TYPE,
        )
    elif method_name == "destroy":
        return ClassMethodDecl(
            classname=class_name,
            method_name=method_name,
            original_func_decl=func,
            type=ClassMethodDecl.DESTRUCTOR_TYPE,
        )
    else:
        if func.in_params[0].name == "self":
            return ClassMethodDecl(
                classname=class_name,
                method_name=method_name,
                original_func_decl=func,
            )
        else:
            return ClassMethodDecl(
                classname=class_name,
                method_name=method_name,
                original_func_decl=func,
                type=ClassMethodDecl.STATIC_METHOD_TYPE,
            )


def translate_arg_from_python_to_c(arg_name: str, param: FuncParameter) -> str:
    from pycparser.c_ast import PtrDecl

    arg_type = param.type
    if param.name == "self":
        assert arg_name == "self"
        return f"self._handle"
    elif param.is_arr_param():
        assert isinstance(arg_type.type, PtrDecl)
        return f"{arg_name}, len({arg_name})"
    elif param.is_cstr_param():
        return f'{arg_name}.encode("utf-8")'
    elif param.is_in_param():
        return arg_name
    elif param.is_out_param():
        assert isinstance(arg_type.type, PtrDecl)
        return f"ctypes.byref({arg_name})"
    else:
        raise ValueError(f"Unknown type: {arg_type}")


def generate_func_def_code_from_func_decl(
    func_def_name: str, original_func: FuncDecl, fp, indent: int, tab: str = None
):
    tab = tab or " " * 4

    def fp_write(content: str):
        fp.write(tab * indent)
        fp.write(content)

    in_params = original_func.in_params
    out_params = original_func.out_params
    # Func def
    in_params = [
        in_params[i]
        for i in range(len(in_params))
        if i == 0 or not in_params[i - 1].is_arr_param()
    ]
    fp_write(
        f"def {func_def_name}("
        + ", ".join([param.name for param in in_params])
        + "):\n"
    )
    # Func body
    args = []
    args.extend(
        [translate_arg_from_python_to_c(param.name, param) for param in in_params]
    )
    args.extend(
        [translate_arg_from_python_to_c(param.name, param) for param in out_params]
    )
    for param in in_params:
        if param.is_arr_param():
            fp_write(
                f"{tab}{param.name} = ({translate_c_type_to_ctypes_type(param.type, exclude_ptr_levels=1)} * len({param.name}))(*{param.name})\n"
            )
    for param in out_params:
        fp_write(
            f"{tab}{param.name} = {translate_c_type_to_ctypes_type(param.type, exclude_ptr_levels=1)}()\n"
        )
    fp_write(f"{tab}ret = taichi_ccore.{original_func.name}(" + ", ".join(args) + ")\n")

    # Process ret (error code)
    assert original_func.ret_type.type.names[0] == "int"
    # TODO: Translate error code to exception
    fp_write(f"{tab}if ret != 0:\n")
    fp_write(f'{tab*2}raise RuntimeError(f"Call failed with error code {{ret}}")\n')

    # Return values
    if len(out_params) > 0:
        fp_write(f"{tab}return (\n")
        fp_write(",\n".join([f"{tab*2}{param.name}.value" for param in out_params]))
        fp_write("\n")
        fp_write(f"{tab})\n")
    fp_write("\n")


def generate_py_module_from_exports_header(dirname: str, header: ExportsHeader):
    COMMENT_HEADER = """# This file is auto-generated by misc/exports_to_py.py
# DO NOT edit this file manually!
# To regenerate this file, run:
#     python misc/exports_to_py.py
"""

    IMPORTS = """import ctypes
from .ccore import taichi_ccore
    """

    os.makedirs(dirname, exist_ok=True)

    # Generate enums (enum0.py, enum1.py, ...)
    for enum in header.enums:
        with open(os.path.join(dirname, f"{enum.name}.py"), "w") as f:
            f.write(COMMENT_HEADER)
            f.write("\n")
            f.write(f'# Enum "{enum.name}"\n')
            for name, value in enum.values.items():
                f.write(f"{name} = {value}\n")

    # Generate class CCore (ccore.py, to load DLL, set argtypes & restype, etc.)
    with open(os.path.join(dirname, "ccore.py"), "w") as f:
        exported_functions_def = (
            "EXPORTED_FUNCTIONS = (\n"
            + ",\n".join(
                [
                    f'    ("{func.name}", [{", ".join([f"{translate_c_type_to_ctypes_type(param.type)}" for param in func.in_params])}, {", ".join([f"{translate_c_type_to_ctypes_type(param.type)}" for param in func.out_params])}], {translate_c_type_to_ctypes_type(func.ret_type)})'
                    for func in header.funcs
                ]
            )
            + "\n)"
        )
        f.write(COMMENT_HEADER)
        f.write("\n")
        f.write(
            CCORE_PYTHON_FILE_FORMAT.format(exported_functions=exported_functions_def)
        )

    # Generate Python class from exported functions (class0.py, class1.py, ...)
    classes: Mapping[str, ClassDecl] = {}
    global_functions: List[FuncDecl] = []
    for func in header.funcs:
        fn = translate_func_decl(func)
        if isinstance(fn, FuncDecl):
            global_functions.append(fn)
        elif isinstance(fn, ClassMethodDecl):
            class_name = fn.classname
            if class_name not in classes:
                classes[class_name] = ClassDecl(name=class_name, methods={})
            classes[class_name].methods[fn.method_name] = fn

    for class_name, class_decl in classes.items():
        with open(os.path.join(dirname, f"{class_name}.py"), "w") as f:
            f.write(COMMENT_HEADER)
            f.write("\n")
            f.write(IMPORTS)
            f.write("\n")
            f.write(f"# Class {class_name}\n")
            f.write(f"class {class_name}:\n")
            f.write("    def __init__(self, *args):\n")
            f.write("        self._handle = self.create(*args)\n")
            f.write("\n")
            f.write("    def __del__(self):\n")
            f.write("        self.destroy()\n")
            f.write("\n")
            f.write("    def get_handle(self):\n")
            f.write("        return self._handle\n")
            f.write("\n")
            for method_name, method_decl in class_decl.methods.items():
                if method_decl.type in (
                    ClassMethodDecl.CONSTRACTOR_TYPE,
                    ClassMethodDecl.STATIC_METHOD_TYPE,
                ):
                    f.write(f"    @staticmethod\n")
                generate_func_def_code_from_func_decl(
                    func_def_name=method_name,
                    original_func=method_decl.original_func_decl,
                    fp=f,
                    indent=1,
                    tab=" " * 4,
                )

    with open(os.path.join(dirname, "global_functions.py"), "w") as f:
        f.write(COMMENT_HEADER)
        f.write("\n")
        f.write(IMPORTS)
        f.write("\n")
        for func in global_functions:
            generate_func_def_code_from_func_decl(
                func_def_name=func.name.replace("tie_G_", ""),
                original_func=func,
                fp=f,
                indent=0,
                tab=" " * 4,
            )

    # Generate __init__.py
    with open(os.path.join(dirname, "__init__.py"), "w") as f:
        # Some comments
        f.write(COMMENT_HEADER)
        f.write("\n")
        # Import: from enum0, enum1, ... import *
        for enum in header.enums:
            f.write(f"from .{enum.name} import *\n")
        f.write("\n")
        # Import: from class0, class1, ... import *
        for class_name in classes.keys():
            f.write(f"from .{class_name} import *\n")
        f.write("\n")
        # Import: from global_functions import *
        f.write("from .global_functions import *\n")
        f.write("\n")


__all__ = ["generate_py_module_from_exports_header"]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <exports.h> <output_dir>")
        sys.exit(1)
    exports_header = parse_exports_header(sys.argv[1])
    generate_py_module_from_exports_header(sys.argv[2], exports_header)
