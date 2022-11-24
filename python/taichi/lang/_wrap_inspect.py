import atexit
import inspect
import os
import tempfile

import sourceinspect

_builtin_getfile = inspect.getfile
_builtin_findsource = inspect.findsource

use_sourceinspect = int(os.getenv('USE_SOURCEINSPECT', 0)) == 1


def _find_source_with_custom_getfile_func(func, obj):
    inspect.getfile = func  # replace with our custom func
    source = inspect.findsource(obj)
    inspect.getfile = _builtin_getfile  # restore
    return source


def _blender_get_text_name(filename: str):
    if filename.startswith(os.path.sep) and filename.count(os.path.sep) == 1:
        return filename[1:]  # "/Text.001" --> "Text.001"

    index = filename.rfind('.blend' + os.path.sep)
    if index != -1:
        return filename[index + 7:]  # "hello.blend/test.py" --> "test.py"

    return None


def _blender_findsource(obj):
    try:
        import bpy  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise IOError('Not in Blender environment!')

    filename = _builtin_getfile(obj)
    text_name = _blender_get_text_name(filename)
    if text_name is None:
        raise IOError(
            'Object `{obj.__name__}` is not defined in a .blend file!')

    lines = bpy.data.texts[text_name].as_string()
    # Now we have found the filename and code lines.
    # We first check if they are already cached, to avoid file io in each query.
    try:
        filename = _blender_findsource._saved_inspect_cache[lines]  # pylint: disable=no-member
    except KeyError:
        fd, filename = tempfile.mkstemp(prefix='_Blender_',
                                        suffix=f'_{text_name}.py')
        os.close(fd)

        with open(filename, 'w') as f:
            f.write(lines)

        _blender_findsource._saved_inspect_cache[lines] = filename  # pylint: disable=no-member
        atexit.register(os.unlink, filename)  # Remove file when program exits

    def wrapped_getfile(ob):
        if id(ob) == id(obj):
            return filename

        return _builtin_getfile(ob)

    return _find_source_with_custom_getfile_func(wrapped_getfile, obj)


_blender_findsource._saved_inspect_cache = {}


def _Python_IPython_findsource(obj):
    try:
        # In Python and IPython the builtin findsource would suffice in most cases
        return _builtin_findsource(obj)
    except IOError:
        # Except that the cell has a magic command like %%time or %%timeit
        # In this case the filename returned by getfile is wrong
        filename = _builtin_getfile(obj)
        if (filename in {"<timed exec>", "<magic-timeit>"}):
            try:
                ip = get_ipython()
                if ip is not None:
                    session_id = ip.history_manager.get_last_session_id()
                    fd, filename = tempfile.mkstemp(prefix='_IPython_',
                                                    suffix=f'_{session_id}.py')
                    os.close(fd)
                    # The latest lines of code are stored in this file
                    lines = ip.history_manager._i00

                    # Remove the magic command (and spaces/sep around it) before saving to a file
                    index = lines.find("%time")
                    lines_stripped = lines[index:]
                    lines_stripped = lines_stripped.split(maxsplit=1)[1]

                    with open(filename, 'w') as f:
                        f.write(lines_stripped)

                    atexit.register(
                        os.unlink,
                        filename)  # Remove the file after the program exits
                    func = lambda obj: filename
                    return _find_source_with_custom_getfile_func(func, obj)

            except:
                pass
        raise IOError(f"Cannot find source code for Object: {obj}")


def _custom_findsource(obj):
    try:
        return _Python_IPython_findsource(obj)
    except IOError:
        try:
            return _blender_findsource(obj)
        except:
            raise IOError(f"Cannot find source code for Object: {obj} ")


class _InspectContextManager:
    def __enter__(self):
        inspect.findsource = _custom_findsource
        return self

    def __exit__(self, *_):
        inspect.findsource = _builtin_findsource


def getsourcelines(obj):
    if use_sourceinspect:
        return sourceinspect.getsourcelines(obj)

    try:
        with _InspectContextManager():
            return inspect.getsourcelines(obj)
    except:
        raise IOError(f"Cannot get the source lines of {obj}. \
            You can try setting `USE_SOURCEINSPECT=1` in the enrionment variables \
                or insert the lines `os.environ['USE_SOURCEINSPECT'] = 1` \
                    at the beginning of the source file. Please report an issue to help us \
                        fix the problem: https://github.com/taichi-dev/taichi/issues if you see this message"
                      )


def getsourcefile(obj):
    if use_sourceinspect:
        return sourceinspect.getsourcefile(obj)

    try:
        with _InspectContextManager():
            ret = inspect.getsourcefile(obj)
            if ret is None:
                ret = inspect.getfile(obj)
            return ret
    except:
        raise IOError(f"Cannot get the source file of {obj}. \
            You can try setting `USE_SOURCEINSPECT=1` in the enrionment variables \
                or insert the lines `os.environ['USE_SOURCEINSPECT'] = 1` \
                    at the beginning of the source file. Please report an issue to help us \
                        fix the problem: https://github.com/taichi-dev/taichi/issues if you see this message"
                      )


__all__ = ['getsourcelines', 'getsourcefile']
