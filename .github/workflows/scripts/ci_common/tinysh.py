# -*- coding: utf-8 -*-

import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

from .escapes import escape_codes

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

P = escape_codes['bold_purple']
N = escape_codes['reset']


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

        overlay = {}
        for v in ENVIRON_STACK:
            overlay.update(v)

        args = prefixes + args
        cmd = ' '.join([quote(v) for v in args])

        print(f'{P}:: RUN {cmd}{N}', file=sys.stderr, flush=True)
        if overlay:
            print(f'{P}>> WITH ADDITIONAL ENVS:{N}',
                  file=sys.stderr,
                  flush=True)
            for k, v in overlay.items():
                print(f'{P}       {k}={v}{N}', file=sys.stderr, flush=True)

        env = os.environ.copy()
        env.update(overlay)

        exe = shutil.which(args[0])
        proc = subprocess.Popen(args, executable=exe, env=env)
        code = proc.wait()
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
bash = sh.bash
start = sh.start.bake('/wait')
