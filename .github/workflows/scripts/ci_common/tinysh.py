# -*- coding: utf-8 -*-

# -- stdlib --
from typing import Sequence, Any
import os
import platform

# -- third party --
# -- own --

# -- code --

# A minimal and naive imitiation of the sh library, which can work on Windows.
# NOT written as a general purpose library, wild assumptions are made.

IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    import mslex
    quote = mslex.quote
else:
    import shlex
    quote = shlex.quote


class CommandFailed(Exception):
    def __init__(self, cmd, code):
        self.cmd = cmd
        self.code = code

    def __str__(self):
        return f'Command {self.cmd} failed with code {self.code}'


class Command:
    def __init__(self, *args: Sequence[str]):
        self.args = list(map(str, args))

    def __getattribute__(self, name: str) -> Any:
        if name in ('args', 'bake') or name.startswith('__'):
            return object.__getattribute__(self, name)

        return self.bake(name)

    def bake(self, *moreargs: Sequence[str]) -> 'Command':
        args = object.__getattribute__(self, 'args')
        cls = object.__getattribute__(self, '__class__')
        return cls(*args, *moreargs)

    def __call__(self, *moreargs: Sequence[str]) -> None:
        args = object.__getattribute__(self, 'args')
        args = args + list(map(str, moreargs))
        cmd = ' '.join(map(quote, args))
        code = os.system(cmd)
        if code:
            raise CommandFailed(cmd, code)

    def __repr__(self) -> str:
        return f"<Command '{shlex.join(self.args)}'>"


def sudo(cmd: Command) -> Command:
    if IS_WINDOWS:
        return cmd
    else:
        return Command('sudo', *cmd.args)


sh = Command()
git = sh.git
# Use setup_python !
# python = sh.bake(sys.executable)
# pip = python.bake('-m', 'pip')
sccache = sh.sccache
tar = sh.tar
