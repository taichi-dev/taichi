'''
Fake a colorama (if not installed)

Q: Why I name this after taichi.color instead of taichi.colorama?
A: To prevent possible self-import on `import colorama` :)
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

    class DummyStyle:
        def __getattribute__(self, x):
            return ''

    def init():
        pass

    Fore = DummyStyle()
    Back = DummyStyle()
    Style = DummyStyle()

    import platform
    if platform.system() != 'Windows':

        class AnsiCodes:
            BLACK = 0
            BLUE = 4
            CYAN = 6
            GREEN = 2
            LIGHTBLACK_EX = 60
            LIGHTBLUE_EX = 64
            LIGHTCYAN_EX = 66
            LIGHTGREEN_EX = 62
            LIGHTMAGENTA_EX = 65
            LIGHTRED_EX = 61
            LIGHTWHITE_EX = 67
            LIGHTYELLOW_EX = 63
            MAGENTA = 5
            RED = 1
            RESET = 9
            WHITE = 7
            YELLOW = 3

        class AnsiStyle:
            BRIGHT = 1
            DIM = 2
            NORMAL = 22

        class AnsiSeq:
            def __init__(self, num):
                self.num = num

            def __str__(self):
                return f'\033[{self.num}m'

            __repr__ = __str__

        class AnsiFore:
            def __getattr__(self, key):
                return AnsiSeq(getattr(AnsiCodes, key) + 30)

        class AnsiBack:
            def __getattr__(self, key):
                return AnsiSeq(getattr(AnsiCodes, key) + 30)

        Fore = AnsiFore()
        Back = AnsiBack()
        Style = AnsiStyle()
