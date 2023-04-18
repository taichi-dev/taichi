# Taichi's custom inspect module.
# This module is used by Taichi's ast transformer to parse the source code.
# Currently this module is aimed for working in the following modes:
# 1. Usual Python/IPython mode, e.g. python script.py
#    In this case we mainly rely on the built-in `inspect` module, except
#    we need some hacks when we are in IPython mode and there is a cell magic.
# 2. Blender's scripting mode, e.g. Users write Taichi code in the scripting
#    window in Blender and press the run button. In this case we need to
#    retrieve the source using Blender's `bpy.data.texts` and write it to a temp
#    file so that the inspect module can parse.
# 3. The interactive shell mode, e.g. Users directly type their code in the
#    interactive shell. In this case we use `dill` to get the source.
#
# NB: Running Taichi in other modes are likely not supported.

import atexit
import inspect
import os
import tempfile

import dill

_builtin_getfile = inspect.getfile
_builtin_findsource = inspect.findsource


def _find_source_with_custom_getfile_func(func, obj):
    """Use a custom function `func` to replace inspect's `getfile`, return the
    source found by the new routine and restore the original `getfile` back.
    """
    inspect.getfile = func  # replace with our custom func
    source = inspect.findsource(obj)
    inspect.getfile = _builtin_getfile  # restore
    return source


def _blender_get_text_name(filename: str):
    """Extract filename from path in the Blender mode."""
    # In Blender's scripting mode, unsaved files are named
    # like `/Text`, `/Text.001`, `/test.py`, etc.
    # We simply remove this path seperator.
    if filename.startswith(os.path.sep) and filename.count(os.path.sep) == 1:
        return filename[1:]  # "/Text.001" --> "Text.001"

    # Saved text files are named like `some-path/xxx.blend/Text` or
    # `some-path/xxx.blend/test.py`
    # We drop the path and extract the filename with extension.
    index = filename.rfind(".blend" + os.path.sep)
    if index != -1:
        return filename[index + 7 :]  # "xxx.blend/test.py" --> "test.py"

    return None


def _blender_findsource(obj):
    try:
        import bpy  # pylint: disable=import-outside-toplevel
    except:
        raise ImportError("Not in Blender environment!")

    # Inspect's built-in `getfile` returns the filename like
    # `/Text`, `/Text.001`, `some-path/xxx.blend/test.py`
    # This filename may not be a full valid path.
    filename = _builtin_getfile(obj)
    # Extract the text name without path
    text_name = _blender_get_text_name(filename)
    if text_name is None:
        raise IOError("Object `{obj.__name__}` is not defined in a .blend file!")
    # Get the lines of code via text_name
    lines = bpy.data.texts[text_name].as_string()
    # Now we have found the lines of code.
    # We first check if they are already cached, to avoid file io in each query.
    try:
        filename = _blender_findsource._saved_inspect_cache[lines]  # pylint: disable=no-member
    except KeyError:
        # Save the code to a valid path.
        fd, filename = tempfile.mkstemp(prefix="_Blender_", suffix=f"_{text_name}.py")
        os.close(fd)

        with open(filename, "w") as f:
            f.write(lines)

        _blender_findsource._saved_inspect_cache[lines] = filename  # pylint: disable=no-member
        atexit.register(os.unlink, filename)  # Remove file when program exits

    # Our custom getfile function
    def wrapped_getfile(ob):
        if id(ob) == id(obj):
            return filename

        return _builtin_getfile(ob)

    return _find_source_with_custom_getfile_func(wrapped_getfile, obj)


_blender_findsource._saved_inspect_cache = {}


def _Python_IPython_findsource(obj):
    try:
        # In Python and IPython the builtin inspect would suffice in most cases
        return _builtin_findsource(obj)
    except IOError:
        # Except that the cell has a magic command like %%time or %%timeit
        # In this case the filename returned by the built-in's getfile is wrong,
        # it becomes something like `<timed exec>` or `<magic-timeit>`.
        filename = _builtin_getfile(obj)
        if filename in {"<timed exec>", "<magic-timeit>"}:
            try:
                ip = get_ipython()
                if ip is not None:
                    # So we are in IPython's cell magic
                    session_id = ip.history_manager.get_last_session_id()
                    fd, filename = tempfile.mkstemp(prefix="_IPython_", suffix=f"_{session_id}.py")
                    os.close(fd)
                    # The latest lines of code can be retrived from here
                    lines = ip.history_manager._i00

                    # `lines` is a string that also contains the cell magic
                    # command, we need to remove the magic command
                    # (and spaces/sep around it) to obtain a valid Python code
                    # snippet before saving it to a file
                    index = lines.find("%time")
                    lines_stripped = lines[index:]
                    lines_stripped = lines_stripped.split(maxsplit=1)[1]

                    with open(filename, "w") as f:
                        f.write(lines_stripped)

                    atexit.register(os.unlink, filename)  # Remove the file after the program exits
                    func = lambda obj: filename
                    return _find_source_with_custom_getfile_func(func, obj)

            except ImportError:
                pass
        raise IOError(
            f"Cannot find source code for Object: {obj}, it's likely \
you are not running Taichi from command line or IPython."
        )


def _REPL_findsource(obj):
    """Findsource in the interactive shell mode."""
    return dill.source.findsource(obj)


def _custom_findsource(obj):
    try:
        return _Python_IPython_findsource(obj)
    except IOError:
        try:
            return _REPL_findsource(obj)
        except:
            try:
                return _blender_findsource(obj)
            except:
                raise IOError(
                    f"Cannot find source code for Object: {obj}, this \
is possibly because of you are running Taichi in an environment that Taichi's own \
inspect module cannot find the source. Please report an issue to help us fix: \
https://github.com/taichi-dev/taichi/issues"
                )


class _InspectContextManager:
    def __enter__(self):
        inspect.findsource = _custom_findsource
        return self

    def __exit__(self, *_):
        inspect.findsource = _builtin_findsource


def getsourcelines(obj):
    with _InspectContextManager():
        return inspect.getsourcelines(obj)


def getsourcefile(obj):
    with _InspectContextManager():
        ret = inspect.getsourcefile(obj)
        if ret is None:
            ret = inspect.getfile(obj)
        return ret


__all__ = ["getsourcelines", "getsourcefile"]
