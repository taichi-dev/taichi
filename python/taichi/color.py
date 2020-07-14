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

            BRIGHT = 1
            DIM = 2
            NORMAL = 22
            RESET_ALL = 0

            def __getattr__(self, key):
                return ''

        Codes = AnsiCodes()

        class AnsiSeq:
            def __init__(self, num):
                self.num = num

            def __str__(self):
                if self.num > 60:
                    return f'\033[1;{self.num - 60}m'
                else:
                    return f'\033[{self.num}m'

            __repr__ = __str__

        class AnsiStyle:
            def __getattr__(self, key):
                return str(AnsiSeq(getattr(Codes, key) + 0))

        class AnsiFore:
            def __getattr__(self, key):
                return str(AnsiSeq(getattr(Codes, key) + 30))

        class AnsiBack:
            def __getattr__(self, key):
                return str(AnsiSeq(getattr(Codes, key) + 40))

        Fore = AnsiFore()
        Back = AnsiBack()
        Style = AnsiStyle()
