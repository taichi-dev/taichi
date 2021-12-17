---
sidebar_position: 1
---

# Contribution guidelines



Thank you for your interest in contributing to Taichi. Taichi was born as an academic research project. Though we are working hard to improve its code quality, Taichi has a long way to go to become a mature, large-scale engineering project. This is also why we decided to open source Taichi from the very beginning: We rely on our community to help Taichi evolve and thrive. From document updates, bug fix, to feature implementation, wherever you spot an issue, you are very welcome to file a PR (pull request) with us!:-)

Centered around the common process of taking on an issue, testing, and making a corresponding PR, this document provides guidelines, tips, and major considerations for Taichi's contributors. We highly recommend that you spend some time familiarizing yourself with this contribution guide before contributing to Taichi.

## General guidelines and tips

This section provides some general guidelines for the Taichi community and tips that we find practically useful.

### Be pragmatic & no overkills

Always use straightforward (sometimes even brute-force) solutions: Complicated code usually suggests a lack of design or over-engineering.

> "There are two ways of constructing a software design: One way is to make it so simple that there are obviously no deficiencies, and the other way is to make it so complicated that there are no obvious deficiencies. *The first method is far more difficult*." — [C.A.R. Hoare](https://en.wikipedia.org/wiki/Tony_Hoare)

### Juxtapose pros and cons

When it comes to making a design decision, weigh up its pros and cons. A design is *good to go,* so long as its advantages outweigh its disadvantages.

> "Perfection (in design) is achieved not when there is nothing more to add, but rather when there is nothing more to take away." [Antoine de Saint-Exupéry](https://en.wikipedia.org/wiki/The_Cathedral_and_the_Bazaar)

### Keep your changesets small

When making a PR, keep your changesets small so that sources of bugs can be easily identified.

### Communicate effectively

Our ultimate goal is to build a sustainable, prosperous Taichi community, and effective communication is the cornerstone of that goal. Following are tips that may contribute to effective communication:

- Concise:
  - The message behind your words outweighs the number of your words. Use as few words as possible to drive your point home.
  - Use tables, figures, and lists where possible.

- Professional:
  - Read twice before you post: Would your point get across with your words?
  - Use a spell checker, such as [Grammarly](https://app.grammarly.com/), to improve your writing in terms of grammar, style, and tone.

- Constructive and courteous: Base your feedback and discussions on facts, *NOT* on personal feelings.
  - Acceptable: *"This design could be confusing to new Taichi users. If it were designed this way, it could..."*
  - Undesirable: *"This design is terrible."*

## What can you contribute

<!-- Todo: Add more information as to where to find the corresponding sources. -->

We welcome all kinds of contributions, including but not limited to:

- Fixing a bug
- Proposing and implementing new features
- Improving or refactoring an existing document
- Suggesting more friendly error messages
- Adding new test cases and examples (demos)
- Posting blog articles and tutorials
- Enhancing compiler performance
- Minor updates to documentation, codes, or annotations.

## Take over an issue

Except for minor updates, most PRs start from a developer taking over an issue. This section provides some corresponding tips and best practices.

### Where to find issues for starters

| Issue Tag                                                    | Description               | Target developer                               |
| ------------------------------------------------------------ | ------------------------- | ---------------------------------------------- |
| [good first issue](https://github.com/taichi-dev/taichi/issues?q=is:open+is:issue+label:"good+first+issue") |                           | Developers new to Taichi                       |
| [welcome contribution](https://github.com/taichi-dev/taichi/issues?q=is:open+is:issue+label:"welcome+contribution") | Slightly more challenging | Developers who wish to dive deeper into Taichi |

### Best practices

- When you plan to take over an issue:
  - **Best practice**: Leave a message claiming that you are working on it.
  - **Goal**: Avoid unnecessary repeated work.
  - **Example**: *"I know how to fix this and would like to help."*

- After you take over an issue:

  - **Best practice**:
    1. Briefly describe how your plan to handle it (if no solution has been provided).
    2. Hold off until a core developer responds to your action plan.

  - **Goal**: Keep your implementation neat and effective.
  - **Example**:  See [#2610](https://github.com/taichi-dev/taichi/issues/2610).

## References for documentation updates

- Documentation source files are hosted under [docs/](https://github.com/taichi-dev/taichi/blob/master/docs/).
- We use GitHub Flavored Markdown (GFM) and [Docusaurus](https://docusaurus.io/) to build our documentation site. For information on the supported Markdown syntax, see the  [Documentation Writing Guide](./doc_writing).
- When it comes to writing, we adhere to the [Google Developer Documentation Style Guide](https://developers.google.com/style/).
- For instruction on setting up a local server and previewing your updated documentation in real-time, see the [Local Development](https://github.com/taichi-dev/docs.taichi.graphics#local-development).

## Add test cases for testing your local changes

We highly recommend that you write your own test cases to cover corner cases for your codes before filing a PR.

- To write a Python test case, see the [Workflow for writing a Python test](./write_test).
- To write a C++ test case, see the [Workflow for writing a CPP test](./writing_cpp_tests).

## Conduct formatting and integration tests locally

We recommend that you complete code formatting and integration tests on your local computer before filing a PR.

## Enforce code style

1. Ensure that you have installed `clang-format-10`.
2. Ensure that you have installed `yapf v0.31.0`.
3. Re-format your code style:

```
python misc/code_format.py
```

<details>
<summary><font color="#006284"> What if I didn't format my code style locally? </font></summary>

1. Have your reviewer leave a comment `/format` in your PR to enable GitHub Actions. See [#2481](https://github.com/taichi-dev/taichi/pull/2481).

  *[Taichi Gardener](https://github.com/taichi-gardener)* *automatically pushes a commit to your branch to format your code.*

2. If you wish to submit more changes after someone leaves the `/format` comment, ensure that your branch is up to date with your remote counterpart.

</details>

<!-- Todo: Make this a reusable fragment. -->

> For more style information for your C++ code, see [our C++ style](./cpp_style).

## Run integration tests

To run all the C++ and Python tests: Run `python tests/run_tests.py` to run all the C++ and Python tests.

**Syntax:**

- `-v`: Verbose output.

- `-t <threads>`: Set a custom number of threads for parallel testing.

- `-a <arch(s)>`: Test only the specified backends (separated by comma).

- `-s`: Original output from the tests.

For example, `python tests/run_tests.py -v -t3 -a cpu,metal -s`

**Syntax:**

`<filename(s)>`: Run test cases in specified files only (separated by comma).

For example, `python tests/run_tests.py numpy_io` runs all tests in [tests/python/test_numpy_io.py](https://github.com/taichi-dev/taichi/blob/master/tests/python/test_numpy_io.py).

**Syntax:**

`-k <key>`: Run only the tests that match the specified keys (supports expression in a key string).

For example, `python tests/run_tests.py linalg -k "cross or diag"` runs `test_cross()` and `test_diag()` in [tests/python/test_linalg.py](https://github.com/taichi-dev/taichi/blob/master/tests/python/test_linalg.py).



For more options, see `python tests/run_tests.py -h`.



>  We have both Python and C++ test cases, but C++ test cases are disabled by default. To enable C++ test cases:
>
>  1. Build Taichi from source using the `python setup.py develop` command.
>  2. Set `TAICHI_CMAKE_ARGS="-DTI_BUILD_TESTS:BOOL=ON"`.

## **File a pull request (PR)**

- PRs with small changesets are preferred.
  - A PR should ideally address only one issue.
    - It is fine to include off-topic **trivial** refactoring such as typo fixes;
    - The reviewer is qualified to request you to remove off-topic **non-trivial** changes in your PR.
  - When implementing a complex feature, consider breaking it down into small PRs to keep a more detailed development history and to interact with core developers more frequently.
  - PR titles should be short sentences describing the changes and following [certain format](./contributor_guide#pr-title-format-and-tags).
  - In the description of a PR, it will be nice to link relevant GitHub issues (e.g. `fixes #issue_number`) or provide a little context on the motivation.

Some important implementation decisions you made in the PR is also helpful.

- If you want early feedback from core developers,
  - Open a PR in [Draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/) state on GitHub to share your progress;
  - Ensure that you @ the corresponding developers in your comments or request reviews from them.
- All PRs should ideally come with corresponding **tests**. See [Testing](./contributor_guide#testing).
- Most PRs, except for those for internal compiler implementations, ought to be accompanied with **document updates**. See [Documentation](./contributor_guide#documentation).
- All PRs must pass **continuous integration tests** before they get merged. See [Using continuous integration](./contributor_guide#using-continuous-integration).
- All PRs must pass **code format checks**. See [Enforcing code style](./contributor_guide#enforcing-code-style).
- Read a great article from Google on [how to have your PR merged quickly](https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html)

[[PDF]](https://github.com/yuanming-hu/public_files/blob/master/graphics/taichi/google_review_comments.pdf)

- PR authors are required to **squash commits on your own**.
- If you are making multiple PRs,
  - Independent PRs should be based on **different** branches forking from `master`;
  - PRs with dependencies should be raised only after all prerequisite PRs are merged into `master`.

### **PR naming conventions**

PR titles will be part of the commit history reflected in the master branch, therefore it is important to keep PR titles easy to understand.

- Prepend at least one tag, such as `[Lang]`, to your PR title:
  - When it comes to multiple tags, separate adjacent tags with exactly one space;
- `[Lang][refactor]` (no space) should be replaced by `[Lang] [refactor]`.
- Capitalize the initial letter of the word immediately after your tag(s):
  -	`[Doc] improve documentation` should be replaced by `[Doc] Improve documentation`;
  -	`[Lang] "ti.sqr(x)" is now deprecated` is fine because `"` is a symbol.

- Do not include back quotes ("\`") in the title of a PR.
- **Good examples**
  - `[Metal] Support bitmasked SNode`,
  - `[Vulkan]ti.atomic_min/max support`, or
  - `[Opt] [ir] Enhanced intra-function optimizations`.

Frequently used tags:

- `[CUDA]`: Backend;
- `[Lang]`: Frontend language features, including syntax sugars;
- `[IR]`: Intermediate representation;
- `[Refactor]`: Code refactoring;
- `[CI]`: CI/CD workflow.

Check out more tags in [misc/prtags.json](https://github.com/taichi-dev/taichi/blob/master/misc/prtags.json). When introducing a new tag, please update [misc/prtags.json](https://github.com/taichi-dev/taichi/blob/master/misc/prtags.json) in the first PR with that tag so that others can follow.

:::note

We do appreciate all kinds of contributions, yet we do not intend to expose the details of every PR to the end-users. Therefore the changelog helps distinguish *what the user should know* from *what the*

*developers are doing*. This is done by **capitalizing PR tags**:

- PRs with visible or notable features to the users should be marked with tags starting with **a capitalized initial**, for example`[Metal]`, `[Vulkan]`, `[IR]`, `[Lang]`. These PRs are [highlighted in the release note](https://github.com/taichi-dev/taichi/blob/master/misc/make_changelog.py) for the end users, therefore it is important that your PR title be self-explanatory.

- Other PRs (underlying development or intermediate implementation) should use tags **in lowercase letters**, for example`[metal]`, `[vulkan]`, `[ir]`, `[lang]`.

- Because of the way the release changelog is generated, there should be **at most one capitalized tag** in a PR title to prevent duplicate PR highlights. For example, `[GUI] [Mac] Support modifier keys` ([#1189](https://github.com/taichi-dev/taichi/pull/1189))is an improper tag choice, and we should have used `[gui] [Mac] Support modifier keys in GUI` instead.

Please capitalize the most relevant tag to your PR.

:::

### **Continuous integration (CI)**

- Continuous Integration (CI) will **build** and **test** your commits in a specific PR in multiple environments.
- CI is triggered each time you push a commit to an open PR.
- A ✔️ on the left-hand side of a commit hash means that CI passes, whilst a ❌ means that CI fails.

### **PR reviewing & merging**

- Please try to follow these tips from Google:
  - [Code Health: Understanding Code In](https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html)
  - [Code Health: Respectful Reviews == Useful](https://testing.googleblog.com/2019/11/code-health-respectful-reviews-useful.html)

- The master branch is required to have a **linear history**. https://idiv-biodiversity.github.io/git-knowledge-base/linear-vs-nonlinear.html

- Ensure that the PR passes all **continuous integration (CI) tests**. See [Using continuous integration](./contributor_guide#using-continuous-integration).

- Ensure that the title follows [PR tag rules](./contributor_guide#pr-title-format-and-tags).



## Frequently asked questions

#### What if I didn't format my code style locally?

1. Have your reviewer leave a comment `/format` in your PR to enable GitHub Actions. See [#2481](https://github.com/taichi-dev/taichi/pull/2481).
   *[Taichi Gardener](https://github.com/taichi-gardener)* *automatically pushes a commit to your branch to format your code.*

2. If you wish to submit more changes after someone leaves the `/format` comment, ensure that your branch is up to date with your remote counterpart.

##  Still have issues?

If you encounter any issue that is not covered here, feel free to report it by asking us on GitHub discussions or by [opening an issue on GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md) and including the details. We are always there to help!

Finally, thanks again for your interest in contributing to Taichi. We look forward to seeing your contributions!
