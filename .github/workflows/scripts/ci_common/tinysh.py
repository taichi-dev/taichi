# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

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


ENVIRON_STACK = []
PREFIX_STACK = []


class Command:
    def __init__(self, *args: str):
        self.args = list((map(str, args)))

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

        prefixes = []
        for v in PREFIX_STACK:
            prefixes.extend(v)

        env = {}
        for v in ENVIRON_STACK:
            env.update(v)

        args = prefixes + args

        code = os.spawnvpe(os.P_WAIT, args[0], args, env)
        if code:
            cmd = ' '.join([quote(v) for v in args])
            raise CommandFailed(cmd, code)

    def __repr__(self) -> str:
        return f"<Command '{shlex.join(self.args)}'>"


@contextmanager
def environ(*envs: Mapping[str, str]):
    '''
    Set command environment variables.
    '''
    global ENVIRON_STACK

    this = {}
    for env in envs:
        this.update(env)

    try:
        ENVIRON_STACK.append(this)
        yield
    finally:
        assert ENVIRON_STACK[-1] is this
        ENVIRON_STACK.pop()


@contextmanager
def prefix(*args: str):
    '''
    Set command prefixes.
    '''
    global PREFIX_STACK

    l = list(map(str, args))

    try:
        PREFIX_STACK.insert(0, l)
        yield
    finally:
        assert PREFIX_STACK[0] is l
        PREFIX_STACK.pop(0)


@contextmanager
def _nop_contextmanager():
    yield


def sudo():
    '''
    Wrap a command with sudo.
    '''
    if IS_WINDOWS:
        return _nop_contextmanager()
    else:
        return prefix('sudo')


sh = Command()
git = sh.git
# Use setup_python !
# python = sh.bake(sys.executable)
# pip = python.bake('-m', 'pip')
sccache = sh.sccache
tar = sh.tar
