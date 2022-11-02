import inspect
import os
import tempfile
import sys


def _blender_get_text_name(filename: str):
    if filename.startswith(os.path.sep) and filename.count(os.path.sep) == 1:
        return filename[1:]

    index = filename.rfind('.blend' + os.path.sep)
    if index != -1:
        return filename[index + 7:]

    return None


def _blender_find_source_text(object):
    try:
        import bpy
    except ImportError:
        raise IOError('Not in Blender environment!')

    filename = inspect.getfile(object)
    text_name = _blender_get_text_name(filename)
    if text_name is None:
        raise IOError('Object `{object.__name__}` is not defined in a .blend file!')

    lines = bpy.data.texts[text_name].as_string()
    return lines, text_name


def _blender_findsource(object):
    lines, text_name = _blender_find_source_text(object)

    try:
        filename = _blender_findsource._saved_inspect_cache[lines]
    except KeyError:
        fd, filename = tempfile.mkstemp(prefix='SI_Blender_', suffix=f'_{text_name}.py')
        os.close(fd)

        with open(filename, 'w') as f:
            f.write(lines)

        _blender_findsource._saved_inspect_cache[lines] = filename

    def wrapped_getfile(obj):
        if id(obj) == id(object):
            return filename

        return inspect._saved_getfile(obj)

    inspect._saved_getfile = inspect.getfile
    inspect.getfile = wrapped_getfile
    ret = inspect.findsource(object)
    inspect.getfile = inspect._saved_getfile
    del inspect._saved_getfile
    return ret


_blender_findsource._saved_inspect_cache = {}


def _Python_IPython_findsource(object):
    try:
        return inspect._saved_findsource(object)
    except IOError:
        filename = inspect.getfile(object)
        if (filename in {"<timed exec>", "<magic-timeit>"}
            and "IPython" in sys.modules
        ):
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                session_id = ip.history_manager.get_last_session_id()
                fd, filename = tempfile.mkstemp(prefix='_IPyhon_', suffix=f'_{session_id}.py')
                os.close(fd)

                lines = ip.history_manager._i00
                index = lines.find("%time")
                lines_stripped = lines[index:]
                lines_stripped = lines_stripped.split(maxsplit=1)[1]

                with open(filename, 'w') as f:
                    f.write(lines_stripped)

                inspect._saved_getfile = inspect.getfile
                inspect.getfile = lambda obj: filename
                ret = inspect.findsource(object)
                inspect.getfile = inspect._saved_getfile
                del inspect._saved_getfile
                return ret
            
        raise IOError(f"Cannot find source code for Object: {object}")


def _custom_findsource(object):
    try:
        return _Python_IPython_findsource(object)
    except IOError:
        try:
            return _blender_findsource(object)
        except:
            raise IOError(f"Cannot find source code for Object: {object} ")


class _InspectContextManager:

    def __enter__(self):
        inspect._saved_findsource = inspect.findsource
        inspect.findsource = _custom_findsource
        return self

    def __exit__(self, *_):
        inspect.findsource = inspect._saved_findsource
        del inspect._saved_findsource


def getsourcelines(object):
    with _InspectContextManager():
        return inspect.getsourcelines(object)


def getsourcefile(object):
    with _InspectContextManager():
        ret = inspect.getsourcefile(object)
        if ret is None:
            try:
                ret = inspect.getfile(object)
            except:
                pass
        return ret


all = ["getsourcelines", "getsourcefile"]