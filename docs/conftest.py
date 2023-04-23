# -*- coding: utf-8 -*-

# -- stdlib --
from dataclasses import dataclass
from functools import wraps
from itertools import count
from re import M
from typing import List, Optional, Dict
import linecache
import sys
import uuid
import warnings

# -- third party --
from pytest import ExceptionInfo
import marko
import matplotlib.pyplot as plt
import pytest
import taichi as ti

# -- own --

# -- code --
warnings.filterwarnings("error", category=DeprecationWarning)

SANE_LANGUAGE_TAGS = {
    "python",
    "c",
    "cpp",
    "cmake",
    "plaintext",
    "text",
    "md",
    "markdown",
    "",
    "shell",
    "bash",
    "sh",
    "mdx-code-block",
    "javascript",
    "js",
    "Gherkin",
}

PRELUDES = {}

PRELUDES[
    "init"
] = """
import taichi as ti
import numpy as np
import math
import random
import torch

ti.init()
i, j, k = (0, 0, 0)
N = 16
M = 8
"""


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


GUI_WINDOW = None


@hook(ti.GUI)
def show(orig, self, *args, **kwargs):
    if not self.running:
        self.close()
        return

    self._frames_remaining -= 1

    return orig(self, *args, *kwargs)


@hook(ti.GUI)
def __init__(orig, self, *args, **kwargs):
    global GUI_WINDOW
    assert not GUI_WINDOW
    orig(self, *args, **kwargs)
    GUI_WINDOW = self


@hook(ti.GUI)
def close(orig, self):
    global GUI_WINDOW
    assert not GUI_WINDOW or self is GUI_WINDOW
    GUI_WINDOW = None
    return orig(self)


GGUI_WINDOW = None


@hook(ti.ui.Window)
def show(orig, self, *args, **kwargs):
    if not self.running:
        self.destroy()
        return

    self._frames_remaining -= 1

    return orig(self, *args, *kwargs)


@hook(ti.ui.Window)
def __init__(orig, self, *args, **kwargs):
    global GGUI_WINDOW
    assert not GGUI_WINDOW
    orig(self, *args, **kwargs)
    GGUI_WINDOW = self


@hook(ti.ui.Window)
def destroy(orig, self):
    global GGUI_WINDOW
    assert not GGUI_WINDOW or self is GGUI_WINDOW
    GGUI_WINDOW = None
    return orig(self)


@hook(plt)
def show(orig):
    return


@hook(plt)
def imshow(orig, img):
    return


_prop_running = property(
    (lambda self: self._frames_remaining > 0),
    (lambda self, v: None),
)

ti.GUI._frames_remaining = 10
ti.GUI.running = _prop_running
ti.ui.Window._frames_remaining = 10
ti.ui.Window.running = _prop_running


def pytest_runtest_teardown(item, nextitem):
    global GUI_WINDOW, GGUI_WINDOW
    GUI_WINDOW and GUI_WINDOW.close()
    GGUI_WINDOW and GGUI_WINDOW.destroy()


@dataclass
class PythonSnippet:
    name: str
    code: str
    skip: Optional[str]
    per_file_preludes: Dict[str, str]
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
            raise ValueError(f"Invalid language tag {bad_tags} in markdown file")

        per_file_preludes = {}

        spec = None
        for name, c in codes:
            if not c.lang == "python":
                continue
            extra = dict((v.split(":", 1) + [None])[:2] for v in c.extra.split())
            code = c.children[0].children
            if "as-prelude" in extra:
                assert "cont" not in extra
                assert "preludes" not in extra
                prelude_name = extra["as-prelude"]
                assert prelude_name not in per_file_preludes, f"Duplicate prelude {prelude_name}"
                per_file_preludes[prelude_name] = code
            elif "cont" in extra:
                assert spec is not None
                spec.code += code
            else:
                if spec is not None:
                    yield MarkdownItem.from_parent(self, name=spec.name, spec=spec)
                preludes = extra.get("preludes")
                if preludes is None:
                    preludes = []
                else:
                    preludes = preludes.split(",")
                spec = PythonSnippet(
                    name=name,
                    code=code,
                    skip=extra.get("skip-ci"),
                    known_error="known-error" in extra,
                    per_file_preludes=per_file_preludes,
                    preludes=preludes,
                )

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
            name = " - ".join(f"[{p}]" for p in path[:end])
            yield f"{name} #{next(counter)}", root
        if not hasattr(root, "children"):
            return

        child_counter = count(1)
        for child in root.children:
            if isinstance(child, marko.inline.InlineElement):
                continue
            if isinstance(child, marko.block.Heading):
                lv = child.level
                path[lv - 1] = "".join(self.extract_text_fragments(child))
                path[lv] = None
                child_counter = count(1)

            yield from self.extract_fenced_code_blocks(child, path, child_counter)

    def extract_text_fragments(self, el):
        if isinstance(el, str):
            yield el
        if not hasattr(el, "children"):
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
            warnings.warn("Known Error, please fix it")
            print(f"::warning:: Known Error: {spec.name}")
            pytest.skip("KnownError")

        preludes = list(spec.preludes)
        if "-init" in preludes:
            preludes.remove("-init")
        else:
            preludes.insert(0, "init")

        snippets = []
        for p in preludes:
            c = spec.per_file_preludes.get(p)
            c = c or PRELUDES.get(p)
            assert c is not None, f"Unknown prelude {p}"
            snippets.append(c)
        snippets.append(spec.code)
        source = "".join(snippets)
        fn = f"<snippet:{uuid.uuid4()}>"
        code = compile(source, fn, "exec")
        linecache.cache[fn] = (
            len(source),
            None,
            [f"{i}\n" for i in source.splitlines()],
            fn,
        )
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
