import os, sys, argparse

from typing import Mapping, List, Union
from pycparser import parse_file
from pycparser.c_ast import Decl as CASTDecl, PtrDecl as CASTPtrDecl, NodeVisitor


UTILS_PYTHON_FILE_CODE = """from .TieError import *
from taichi.lang.exception import TaichiTypeError, TaichiSyntaxError, TaichiIndexError, TaichiAssertionError, TaichiRuntimeError


class TieAPIError(Exception):
    pass

    
TIE_ERROR_TO_PYTHON_EXCEPTION = {
    TIE_ERROR_INVALID_ARGUMENT: TieAPIError,
    TIE_ERROR_INVALID_RETURN_ARG: TieAPIError,
    TIE_ERROR_INVALID_HANDLE: TieAPIError,
    TIE_ERROR_TAICHI_TYPE_ERROR: TaichiTypeError,
    TIE_ERROR_TAICHI_SYNTAX_ERROR: TaichiSyntaxError,
    TIE_ERROR_TAICHI_INDEX_ERROR: TaichiIndexError,
    TIE_ERROR_TAICHI_RUNTIME_ERROR: TaichiRuntimeError,
    TIE_ERROR_TAICHI_ASSERTION_ERROR: TaichiAssertionError,
    TIE_ERROR_OUT_OF_MEMORY: TieAPIError,
    TIE_ERROR_UNKNOWN_CXX_EXCEPTION: TieAPIError,
}


def get_exception_to_throw_if_not_success(ret, last_err, last_err_msg):
    assert ret == 0 or ret == last_err
    if ret != 0:
        assert ret in TIE_ERROR_TO_PYTHON_EXCEPTION
        return TIE_ERROR_TO_PYTHON_EXCEPTION[ret](last_err_msg)
    return None


__all__ = [
    "TieAPIError",
    "get_exception_to_throw_if_not_success",
]
"""


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


def type_exclude_ptr(type, levels: int):
    from pycparser.c_ast import PtrDecl

    if isinstance(type, Type):
        type = type.type

    if levels == 0:
        return type
    if isinstance(type, PtrDecl):
        return type_exclude_ptr(type.type, levels - 1)
    return type


def type_is_cstr(type: Union[Type, CASTPtrDecl]) -> bool:
    from pycparser.c_ast import PtrDecl, TypeDecl, IdentifierType

    if isinstance(type, Type):
        type = type.type

    return (
        isinstance(type, PtrDecl)
        and isinstance(type.type, TypeDecl)
        and isinstance(type.type.type, IdentifierType)
        and type.type.type.names[0] == "char"
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
        return type_is_cstr(self.type)


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
    def __init__(self, printer):
        self._printer = printer
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
        self._printer(f"Found function: {func_name}", verbose_level=2)

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
            self._printer(f"Found enum: {enum_name}", verbose_level=2)


def make_printer(verbose: int):
    def printer(*args, verbose_level=1, **kwargs):
        if verbose >= verbose_level:
            print("Generating Python module from exports.h: ", *args, **kwargs)

    return printer


def parse_exports_header(
    filename: str, cpp_path: str, cpp_args: List[str], printer
) -> ExportsHeader:
    assert filename.endswith(".h")

    printer(f"Parsing {filename} ...")
    ast = parse_file(
        filename,
        use_cpp=True,
        cpp_path=cpp_path,  # '{CMAKE_C_COMPILER}'
        cpp_args=cpp_args,  # ['-E', '-DTI_EXPORTS_TO_PY', '-I{fake_libc_include}']
    )
    printer(f"Parsing {filename} done")

    printer(f"Finding enums and functions in {filename} ...")
    visitor = ExportsHeaderASTVisitor(printer=printer)
    visitor.visit(ast)
    enums, funcs = visitor.enums, visitor.funcs
    printer(f"Found {len(enums)} enums and {len(funcs)} functions in {filename}")
    return ExportsHeader(enums=enums, funcs=funcs)


def translate_c_type_to_ctypes_type(type, exclude_ptr_levels: int = 0) -> str:
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


def translate_ret_from_c_to_python(ret_var_name: str, out_param: FuncParameter) -> str:
    assert out_param.is_out_param()

    value_type = type_exclude_ptr(out_param.type, 1)
    if type_is_cstr(value_type):
        return f'ctypes.string_at({ret_var_name}.value).decode("utf-8")'
    else:
        return f"{ret_var_name}.value"


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
    if func_def_name == "get_last_error":  # NOTE: Avoid infinite recursion
        fp_write(f"{tab}if ret != 0:\n")
        fp_write(
            f'{tab*2}raise RuntimeError(f"Failed to call get_last_error, err={{ret}}")\n'
        )
    else:
        fp_write(
            f"{tab}ex = get_exception_to_throw_if_not_success(ret, *get_last_error())\n"
        )
        fp_write(f"{tab}if ex is not None:\n")
        fp_write(f"{tab*2}raise ex\n")

    # Return values
    if len(out_params) > 0:
        fp_write(f"{tab}return (\n")
        fp_write(
            ",\n".join(
                [
                    f"{tab*2}{translate_ret_from_c_to_python(param.name, param)}"
                    for param in out_params
                ]
            )
        )
        fp_write("\n")
        fp_write(f"{tab})\n")
    fp_write("\n")


def generate_py_module_from_exports_header(
    dirname: str, header: ExportsHeader, printer
):
    COMMENT_HEADER = """# This file is auto-generated by misc/exports_to_py.py
# DO NOT edit this file manually!
# To regenerate this file, run:
#     python misc/exports_to_py.py
"""

    IMPORTS = """import ctypes
from .ccore import taichi_ccore
from .utils import get_exception_to_throw_if_not_success

"""

    os.makedirs(dirname, exist_ok=True)

    # Dump utils.py
    printer(f"Generating {os.path.join(dirname, 'utils.py')} ...")
    with open(os.path.join(dirname, "utils.py"), "w") as f:
        f.write(COMMENT_HEADER)
        f.write("\n")
        f.write(UTILS_PYTHON_FILE_CODE)

    # Generate enums (enum0.py, enum1.py, ...)
    for enum in header.enums:
        printer(f"Generating {os.path.join(dirname, f'{enum.name}.py')} ...")
        with open(os.path.join(dirname, f"{enum.name}.py"), "w") as f:
            f.write(COMMENT_HEADER)
            f.write("\n")
            f.write(f'# Enum "{enum.name}"\n')
            for name, value in enum.values.items():
                f.write(f"{name} = {value}\n")

    # Generate class CCore (ccore.py, to load DLL, set argtypes & restype, etc.)
    printer(f"Generating {os.path.join(dirname, 'ccore.py')} ...")
    with open(os.path.join(dirname, "ccore.py"), "w") as f:
        exported_functions_def = (
            "EXPORTED_FUNCTIONS = (\n"
            + ",\n".join(
                [
                    f'    ("{func.name}", [{", ".join([f"{translate_c_type_to_ctypes_type(param.type)}" for param in (func.in_params + func.out_params)])}], {translate_c_type_to_ctypes_type(func.ret_type)})'
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
        printer(
            f"Generating class {class_name} ({os.path.join(dirname, f'{class_name}.py')}) ..."
        )
        with open(os.path.join(dirname, f"{class_name}.py"), "w") as f:
            f.write(COMMENT_HEADER)
            f.write("\n")
            f.write(IMPORTS)
            f.write("\n")
            f.write(f"from .global_functions import get_last_error\n\n")
            f.write("\n")
            f.write(f"# Class {class_name}\n")
            f.write(f"class {class_name}:\n")
            f.write("    def __init__(self, *args):\n")
            f.write("        self._handle = self.create(*args)\n")
            f.write("\n")
            f.write("    def __del__(self):\n")
            f.write("        try:\n")
            f.write("            self.destroy()\n")
            f.write("        except:\n")
            f.write("            pass\n")
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
            f.write("\n")
            f.write(f"__all__ = ['{class_name}']\n")

    # Generate global functions (global_functions.py)
    printer(
        f"Generating global functions ({os.path.join(dirname, 'global_functions.py')}) ..."
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
        f.write("\n")
        f.write(f"__all__ = [\n")
        f.write(
            ",\n".join(
                [
                    f'    "{func.name.replace("tie_G_", "")}"'
                    for func in global_functions
                ]
            )
        )
        f.write("\n]\n")

    # Generate __init__.py
    printer(f"Generating {os.path.join(dirname, '__init__.py')} ...")
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


__all__ = ["parse_exports_header", "generate_py_module_from_exports_header"]


if __name__ == "__main__":
    # This tool is designed to generate Python module for taichi/exports/exports.h, which is not a general tool
    # Usage: python misc/exports_to_py.py \
    #           --exports-header taichi/exports/exports.h \
    #           --cpp-path ${CMAKE_C_COMPILER} \
    #           --cpp-args "['-E', '-DTI_EXPORTS_TO_PY', '-Iexternal/pycparser/utils/fake_libc_include']" \
    #           --output-dir python/taichi/_lib/exports \
    #           --verbose 2

    parser = argparse.ArgumentParser(
        description=f"Generate Python module from exports.h"
    )

    parser.add_argument(
        "--exports-header",
        dest="exports_header",
        type=str,
        required=True,
        help="Path to exports.h",
    )
    parser.add_argument(
        "--cpp-path",
        dest="cpp_path",
        type=str,
        required=True,
        help="Path to C preprocessor",
    )
    parser.add_argument(
        "--cpp-args",
        dest="cpp_args",
        type=str,
        required=True,
        help="Args to C preprocessor",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        type=int,
        help="Verbose level",
        required=False,
        default=0,  # 0: no output, 1: output, 2: output more details
    )

    args = parser.parse_args()

    cpp_args = eval(args.cpp_args)
    assert isinstance(cpp_args, list)
    printer = make_printer(args.verbose)

    exports_header = parse_exports_header(
        filename=args.exports_header,
        cpp_path=args.cpp_path,
        cpp_args=cpp_args,
        printer=printer,
    )
    generate_py_module_from_exports_header(
        dirname=args.output_dir, header=exports_header, printer=printer
    )
