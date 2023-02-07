import linecache
import sys
import uuid
import warnings
from dataclasses import dataclass
from functools import wraps
from itertools import count
from typing import List, Optional

import marko
import pytest
from pytest import ExceptionInfo

import taichi as ti

SANE_LANGUAGE_TAGS = {
    'python',
    'c',
    'cpp',
    'cmake',
    'plaintext',
    'text',
    'md',
    'markdown',
    '',
    'shell',
    'bash',
    'sh',
    'mdx-code-block',
    'javascript',
    'js',
    'Gherkin',
}

PRELUDES = {}

PRELUDES['init'] = '''
import taichi as ti
import numpy as np
import math
import random
import torch

ti.init()
i, j, k = (0, 0, 0)
N = 16
M = 8

'''


def hook(module, name=None):
    def inner(hooker):
        funcname = name or hooker.__name__
        hookee = getattr(module, funcname)

        @wraps(hookee)
        def real_hooker(*args, **kwargs):
            return hooker(hookee, *args, **kwargs)

        real_hooker.orig = hookee
        setattr(module, funcname, real_hooker)
        return real_hooker

    return inner


@hook(ti.GUI)
def show(orig, self, *args, **kwargs):
    if not self.running:
        self.close()
        return

    self._frames_remaining -= 1

    return orig(self, *args, *kwargs)


@hook(ti.ui.Window)
def show(orig, self, *args, **kwargs):
    if not self.running:
        self.close()
        return

    self._frames_remaining -= 1

    return orig(self, *args, *kwargs)


ti.GUI._frames_remaining = 10
ti.GUI.running = property(lambda self: self._frames_remaining <= 0)
ti.ui.Window._frames_remaining = 10
ti.ui.Window.running = property(lambda self: self._frames_remaining <= 0)


@dataclass
class PythonSnippet:
    name: str
    code: str
    skip: Optional[str]
    known_error: bool = False
    preludes: Optional[List[str]] = None


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".md":
        return MarkdownFile.from_parent(parent, path=file_path)


class MarkdownFile(pytest.File):
    def collect(self):
        doc = marko.parse(self.path.read_text())
        codes = list(self.extract_fenced_code_blocks(doc))
        bad_tags = set(c.lang for _, c in codes) - SANE_LANGUAGE_TAGS
        if bad_tags:
            raise ValueError(
                f"Invalid language tag {bad_tags} in markdown file")

        spec = None
        for name, c in codes:
            if not c.lang == 'python':
                continue
            extra = dict(
                (v.split(':', 1) + [None])[:2] for v in c.extra.split())
            code = c.children[0].children
            if 'cont' in extra:
                assert spec is not None
                spec.code += code
            else:
                if spec is not None:
                    yield MarkdownItem.from_parent(self,
                                                   name=spec.name,
                                                   spec=spec)
                preludes = extra.get('preludes')
                if preludes is None:
                    preludes = ['init']
                else:
                    preludes = preludes.split(',')
                spec = PythonSnippet(name=name,
                                     code=code,
                                     skip=extra.get('skip-ci'),
                                     known_error='known-error' in extra,
                                     preludes=preludes)

        if spec is not None:
            yield MarkdownItem.from_parent(self, name=spec.name, spec=spec)

    def extract_fenced_code_blocks(self, root, path=None, counter=None):
        path = path or [None] * 20
        counter = counter or count(1)
        if root is None:
            return
        if isinstance(root, str):
            return
        if isinstance(root, marko.block.FencedCode):
            end = path.index(None)
            name = ' - '.join(f'[{p}]' for p in path[:end])
            yield f'{name} #{next(counter)}', root
        if not hasattr(root, 'children'):
            return

        child_counter = count(1)
        for child in root.children:
            if isinstance(child, marko.inline.InlineElement):
                continue
            if isinstance(child, marko.block.Heading):
                lv = child.level
                path[lv - 1] = ''.join(self.extract_text_fragments(child))
                path[lv] = None
                child_counter = count(1)

            yield from self.extract_fenced_code_blocks(child, path,
                                                       child_counter)

    def extract_text_fragments(self, el):
        if isinstance(el, str):
            yield el
        if not hasattr(el, 'children'):
            return
        for child in el.children:
            yield from self.extract_text_fragments(child)


class MarkdownItem(pytest.Item):
    def __init__(self, *, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

    def runtest(self):
        spec = self.spec
        if spec.skip is not None:
            pytest.skip(spec.skip)

        if spec.known_error:
            warnings.warn('Known Error, please fix it')
            pytest.skip('KnownError')
            return

        source = [PRELUDES[p] for p in spec.preludes] + [spec.code]
        source = ''.join(source)
        fn = f'<snippet:{uuid.uuid4()}>'
        code = compile(source, fn, 'exec')
        linecache.cache[fn] = (len(source), None,
                               [f'{i}\n' for i in source.splitlines()], fn)
        env = {}
        try:
            exec(code, env, env)
        except Exception:
            excinfo = sys.exc_info()
            raise MarkdownException(excinfo)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, MarkdownException):
            return super().repr_failure(ExceptionInfo(excinfo.value.excinfo))

        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, self.name


class MarkdownException(Exception):
    """Custom exception for error reporting."""
    def __init__(self, excinfo):
        self.excinfo = excinfo
