---
sidebar_position: 1
---

# Contribution guidelines

First of all, thank you for contributing! We welcome all kinds of contributions, including but not limited to

- Bug fixes
- New feature proposals and implementations
- Documentation improvements and translations
- More user-friendly error messages
- New test cases and examples
- Compiler performance enhancements
- High-quality blog posts and tutorials
- Participation in the [Taichi forum](https://forum.taichi.graphics/)
- Introducing Taichi to your friends or simply staring [the
  project on GitHub](https://github.com/taichi-dev/taichi)
- Typo fixes in the documentation, code or comments (please go ahead and
  make a pull request for minor issues like these)

:::tip reminder
Please take some time to familiarize yourself with this contribution guide before opening a pull request.
For more details regarding development of the Taichi compiler, read the [development tips](./development_tips).
:::

## Where to find contribution opportunities

- Issues marked with ["good first
issue"](https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
are great chances for starters.
- Issues marked with ["welcome
  contribution"](https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22welcome+contribution%22)
  are slightly more challenging but still friendly to beginners.

## How to take over an issue

- Please first leave a comment (e.g. _I know how to fix this and would
  like to help!_) on the issue, so that people know someone is already
  working on it. This helps prevent redundant work.
- If no core developer has commented and described a potential
  solution on the issue, please briefly describe your plan, and wait
  for a core developer to reply before you start. This helps keep
  implementations simple and effective.

## High-level guidelines

- Be pragmatic: practically solving problems is our ultimate goal.
- No overkills: always use _easy_ solutions to solve easy problems, so
  that you have time and energy for real hard ones.
- Almost every design decision has pros and cons. A decision is
  *good* if its pros outweigh its cons. Always think about
  both sides.
- Debugging is hard. Changesets should be small so that sources of
  bugs can be easily pinpointed.
- Unit tests and integration tests are our friends.

:::note
"There are two ways of constructing a software design: One way is to
make it so simple that there are obviously no deficiencies, and the
other way is to make it so complicated that there are no obvious
deficiencies. _The first method is far more difficult_."
— [C.A.R. Hoare](https://en.wikipedia.org/wiki/Tony_Hoare)
:::

One thing to keep in mind is that, Taichi was originally born as an
academic research project. This usually means that some parts did not
have the luxury to go through a solid design. While we are always trying
to improve the code quality, it doesn't mean that the project is free
from technical debts. Some places may be confusing or overly
complicated. Whenever you spot one, you are more than welcome to shoot
us a PR! :-)

## Effective communication

A few tips for effective communication in the Taichi community:

- How much information one effectively conveys, is way more important
  than how many words one typed.
- Be constructive. Be polite. Be organized. Be concise.
- Bulleted lists are our friends.
- Proofread before you post: if you are the reader, can you understand
  what you typed?
- If you are not a native speaker, consider using a spell checker such
  as [Grammarly](https://app.grammarly.com/).

Please base your discussion and feedback on facts, and not personal
feelings. It is very important for all of us to maintain a friendly and
blame-free community. Some examples:

:::tip Acceptable :-)
This design could be confusing to new Taichi users.
:::

:::danger Not Acceptable
This design is terrible.
:::

## Making good pull requests (PRs)

- PRs with **small** changesets are preferred.
  - A PR should ideally address **only one issue**.
    - It is fine to include off-topic **trivial** refactoring such as
      typo fixes;
    - The reviewers reserve the right to ask PR authors to remove
      off-topic **non-trivial** changes.
  - When implementing a complex feature, consider breaking it down into
    small PRs to keep a more detailed development history and to
    interact with core developers more frequently.
- PR titles should be short sentences describing the changes and following
  [certain format](./contributor_guide#pr-title-format-and-tags).
- In the description of a PR, it will be nice to link relevant GitHub issues
  (e.g. `fixes #issue_number`) or provide a little context on the motivation.
  Some important implementation decisions you made in the PR is also helpful.
- If you want early feedback from core developers,
  - Open a PR in
    [Draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/)
    state on GitHub to share your progress;
  - Make sure you @ the corresponding developers in the comments or
    request reviews from them.
- All PRs should ideally come with corresponding **tests**. See [Testing](./contributor_guide#testing).
- All PRs should come with **documentation updates**, except for
  internal compiler implementations. See [Documentation](./contributor_guide#documentation).
- All PRs must pass **continuous integration tests** before they get
  merged. See [Using continuous integration](./contributor_guide#using-continuous-integration).
- All PRs must pass **code format checks**. See [Enforcing code style](./contributor_guide#enforcing-code-style).
- Read a great article from Google on [how to have your PR merged
  quickly](https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html).
  [\[PDF\]](https://github.com/yuanming-hu/public_files/blob/master/graphics/taichi/google_review_comments.pdf)
- All commits in a PR will always be **squashed and merged into master
  as a single commit**. However, PR authors **should not squash commits on their own**.
- If you are making multiple PRs,
  - Independent PRs should be based on **different** branches
    forking from `master`;
  - PRs with dependencies should be raised only after all
    prerequisite PRs are merged into `master`.


## PR reviewing & merging

- Please try to follow these tips from Google:
  - [Code Health: Understanding Code In
    Review](https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html);
    [\[PDF\]](https://github.com/yuanming-hu/public_files/blob/master/graphics/taichi/google_understanding_code.pdf)
  - [Code Health: Respectful Reviews == Useful
    Reviews](https://testing.googleblog.com/2019/11/code-health-respectful-reviews-useful.html).
    [\[PDF\]](https://github.com/yuanming-hu/public_files/blob/master/graphics/taichi/google_respectful_reviews.pdf)
- The merger should always **squash and merge** PRs into the master
  branch.
- The master branch is required to have a **linear history**.
- Make sure the PR passes **continuous integration tests**, except for
  cases like documentation updates. See [Using continuous integration](./contributor_guide#using-continuous-integration).
- Make sure the title follows [PR tag rules](./contributor_guide#pr-title-format-and-tags).

## Using continuous integration

- Continuous Integration (CI) will **build** and **test** your
  commits in a PR in multiple environments.
- Currently, Taichi uses [Github Actions](https://github.com/features/actions).
- CI will be triggered every time you push commits to an open PR.
- You can prepend `[skip ci]` to your commit message to avoid
  triggering CI. For example, a commit with the message
  `[skip ci] This commit will not trigger CI` will not trigger CI.
- A tick on the left-hand side of a commit hash means CI passed, while
  a cross means CI failed.

## Enforcing code style

- Locally, you can run `python misc/code_format.py` in the command line to re-format
  code style. Note that you have to install `clang-format-10` and
  `yapf v0.31.0` locally.

- If you don't have these formatting tools locally, feel free to
  leverage GitHub Actions: simply comment `/format` in a PR
  (e.g., [#2481](https://github.com/taichi-dev/taichi/pull/2481#issuecomment-872226701))
  and then [Taichi Gardener](https://github.com/taichi-gardener)
  will automatically push a commit to your branch that formats the code.
  Note if you want to make more changes afterwards, you'll need to
  `git pull` first.

- For your C++ code, please also follow [C++ style](./cpp_style).

## PR title format and tags

PR titles will be part of the commit history reflected in the `master`
branch, therefore it is important to keep PR titles readable.

- Please always prepend **at least one tag** such as `[Lang]` to PR
  titles:
  - When using multiple tags, make sure there is exactly one
    space between tags;
  - For example, `[Lang][refactor]` (no space) should be replaced
    by `[Lang] [refactor]`.
- The first letter of the PR title body should be capitalized:
  - For example, `[Doc] improve documentation` should be replaced by
    `[Doc] Improve documentation`;
  - `[Lang] "ti.sqr(x)" is now deprecated` is fine because `"`
    is a symbol.
- Please do not include back quotes ("`") in PR titles.
- Good examples include `[Metal] Support bitmasked SNode`, `[Vulkan]
  ti.atomic_min/max support`, or `[Opt] [ir] Enhanced intra-function optimizations`.

Frequently used tags:

- `[CPU]`, `[CUDA]`, `[Metal]`, `[Vulkan]`, `[OpenGL]`: backends;
- `[LLVM]`: the LLVM backend shared by CPUs and CUDA;
- `[Lang]`: frontend language features, including syntax sugars;
- `[Std]`: standard library, e.g., `ti.Matrix` and `ti.Vector`;
- `[Sparse]`: sparse computation;
- `[IR]`: intermediate representation;
- `[Opt]`: IR optimization passes;
- `[GUI]`: the built-in GUI system;
- `[Refactor]`: code refactoring;
- `[CLI]`: commandline interfaces, e.g., the `ti` command;
- `[Doc]`: documentation under [docs/](https://github.com/taichi-dev/taichi/blob/master/docs/);
- `[Example]`: examples under [examples/](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/);
- `[Test]`: tests under [tests/](https://github.com/taichi-dev/taichi/blob/master/tests/);
- `[Linux]`: Linux platform;
- `[Android]`: Android platform;
- `[Mac]`: macOS platform;
- `[Windows]`: Windows platform;
- `[Perf]`: performance improvements;
- `[CI]`: CI/CD workflow;
- `[Misc]`: something that doesn't belong to any category, such as
  version bump, reformatting;
- `[Bug]`: bug fixes.

Check out more tags in
 [misc/prtags.json](https://github.com/taichi-dev/taichi/blob/master/misc/prtags.json). When introducing a new tag, please update
 [misc/prtags.json](https://github.com/taichi-dev/taichi/blob/master/misc/prtags.json) in the first PR with that tag, so that people can
  follow.

:::note

We do appreciate all kinds of contributions, yet we should not expose
the title of every PR to end-users. Therefore the changelog will
distinguish *what the user should know* from *what the
developers are doing*. This is done by **capitalizing PR
tags**:

- PRs with visible or notable features to the users should be marked
  with tags starting with **the first letter capitalized**, e.g.,
  `[Metal]`, `[Vulkan]`, `[IR]`, `[Lang]`, `[CLI]`. These PRs will be
  [highlighted in the release note](https://github.com/taichi-dev/taichi/blob/master/misc/make_changelog.py)
  for end-users, therefore it is important to make sure your PR title is
  effective in telling what your PR does.
- Other PRs (underlying development or intermediate implementation)
  should use tags with **everything in lowercase letters**, e.g.,
  `[metal]`, `[vulkan]`, `[ir]`, `[lang]`, `[cli]`.
- Because of the way the release changelog is generated, there
  should be **at most one capitalized tag** in a PR title to prevent
  duplicate PR highlights. For example,
  `[GUI] [Mac] Support modifier keys` ([#1189](https://github.com/taichi-dev/taichi/pull/1189))
  is an improper tag choice, and we
  should have used `[gui] [Mac] Support modifier keys in GUI` instead.
  Please capitalize the tag that is the *most* relevant to the PR.
:::

## Testing

Tests should be added to [tests/](https://github.com/taichi-dev/taichi/blob/master/tests/). We
have both Python tests and C++ tests.

### Python tests

- Use `python tests/run_tests.py` to run all the tests.
- Use `python tests/run_tests.py -v` for verbose outputs.
- Use `python tests/run_tests.py -s` for original output from the tests.
- Use `python tests/run_tests.py -a <arch(s)>` to test only specified backends, e.g.,
  `python tests/run_tests.py -a cuda,metal`.
- Use `python tests/run_tests.py -na <arch(s)>` to test all backends excluding specified ones,
  e.g., `python tests/run_tests.py -na opengl,x64`.
- Use `python tests/run_tests.py <filename(s)>` to run tests in specified files. For example,
  `python tests/run_tests.py numpy_io` will run all tests in [tests/python/test_numpy_io.py](https://github.com/taichi-dev/taichi/blob/master/tests/python/test_numpy_io.py).
- Use `python tests/run_tests.py -k <key>` to run tests that match the specified key. For
  example, `python tests/run_tests.py linalg -k "cross or diag"` will run `test_cross`
  and `test_diag` in [tests/python/test_linalg.py](https://github.com/taichi-dev/taichi/blob/master/tests/python/test_linalg.py).
- Use `python tests/run_tests.py -t <threads>` to set custom number of threads for parallel testing.

For more options, see `python tests/run_tests.py -h`.

For more details on how to write a Python test case, see
[Workflow for writing a Python test](./write_test).

### C++ tests

For more details on C++ tests, see
[Workflow for writing a CPP test](./writing_cpp_tests).

## Documentation

Documentation source files are under [docs/](https://github.com/taichi-dev/taichi/blob/master/docs/) of [**the main Taichi repo**](https://github.com/taichi-dev/taichi).
An automatic service syncs the updated content with our [documentation repo](https://github.com/taichi-dev/docs.taichi.graphics) and deploys the documentation at [the Taichi documentation site](https://docs.taichi.graphics).

We use [Markdown](https://www.markdownguide.org/getting-started/) (.md) to write documentation.
Please see [the documentation writing guide](./doc_writing) for more tips.

To set up a local server and preview your documentation edits in real time,
see instructions for [Local Development](https://github.com/taichi-dev/docs.taichi.graphics#local-development).
