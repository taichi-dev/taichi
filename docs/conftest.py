# content of conftest.py
from dataclasses import dataclass
from itertools import count

import marko
import pytest

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


@dataclass
class PythonSnippet:
    name: str
    code: str


def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".md":
        return MarkdownFile.from_parent(parent, path=file_path)


class MarkdownFile(pytest.File):
    def collect(self):
        doc = marko.parse(self.path.read_text())
        codes = list(self.extract_fenced_code_blocks(doc))
        bad_tags = set(c.lang for name, c in codes) - SANE_LANGUAGE_TAGS
        if bad_tags:
            raise ValueError(
                f"Invalid language tag {bad_tags} in markdown file")

        spec = None
        for name, c in codes:
            if not c.lang == 'python':
                continue
            extra = c.extra.split()
            code = c.children[0].children
            # if 'ci' not in extra:
            #     continue
            if 'cont' in extra:
                assert spec is not None
                spec.code += code
            else:
                if spec is not None:
                    yield MarkdownItem.from_parent(self,
                                                   name=spec.name,
                                                   spec=spec)
                spec = PythonSnippet(name=name, code=code)

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
        return True
        for name, value in sorted(self.spec.items()):
            # Some custom test execution (dumb example follows).
            if name != value:
                raise MarkdownException(self, name, value)

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, MarkdownException):
            return "\n".join([
                "usecase execution failed",
                "   spec failed: {1!r}: {2!r}".format(*excinfo.value.args),
                "   no further details known at this point.",
            ])

    def reportinfo(self):
        return self.path, 0, f"usecase: {self.name}"


class MarkdownException(Exception):
    """Custom exception for error reporting."""
