'''
Fake a colorama (if not installed)
'''

_has_colorama = False
_env_colorama = __import__('os').environ.get('TI_ENABLE_COLORAMA', '1')
if not _env_colorama or int(_env_colorama):
    try:
        import colorama
        _has_colorama = True
    except:
        pass

if _has_colorama:
    from colorama import *

else:

    def init():
        pass

    class _styler:
        def __getattribute__(self, x):
            return ''

    Fore = _styler()
    Back = _styler()
    Style = _styler()
