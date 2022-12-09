"""
Generates a dictionary of ANSI escape codes.

http://en.wikipedia.org/wiki/ANSI_escape_code

Uses colorama as an optional dependency to support color on Windows

---
Copied from colorlog
---
"""

__all__ = ('escape_codes', 'parse_colors')


# Returns escape codes from format codes
def esc(*x):
    return '\033[' + ';'.join(x) + 'm'


# The initial list of escape codes
escape_codes = {
    'reset': esc('0'),
    'bold': esc('01'),
    'thin': esc('02')
}

# The color names
COLORS = [
    'black',
    'red',
    'green',
    'yellow',
    'blue',
    'purple',
    'cyan',
    'white'
]

PREFIXES = [
    # Foreground without prefix
    ('3', ''), ('01;3', 'bold_'), ('02;3', 'thin_'),

    # Foreground with fg_ prefix
    ('3', 'fg_'), ('01;3', 'fg_bold_'), ('02;3', 'fg_thin_'),

    # Background with bg_ prefix - bold/light works differently
    ('4', 'bg_'), ('10', 'bg_bold_'),
]

for prefix, prefix_name in PREFIXES:
    for code, name in enumerate(COLORS):
        escape_codes[prefix_name + name] = esc(prefix + str(code))


def parse_colors(sequence):
    """Return escape codes from a color sequence."""
    return ''.join(escape_codes[n] for n in sequence.split(',') if n)
