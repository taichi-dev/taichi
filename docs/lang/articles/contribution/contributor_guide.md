---
sidebar_position: 1
---

# Contribution Guidelines

Thank you for your interest in contributing to Taichi. Taichi was born as an academic research project. Though we are working hard to improve its code quality, Taichi has a long way to go to become a mature, large-scale engineering project. This is also why we decided to open source Taichi from the very beginning: We rely on our community to help Taichi evolve and thrive. From document updates, bug fix, to feature implementation, wherever you spot an issue, you are very welcome to file a PR (pull request) with us!:-)

Centered around the common process of taking on an issue, testing, and making a corresponding PR, this document provides guidelines, tips, and major considerations for Taichi's contributors. We highly recommend that you spend some time familiarizing yourself with this contribution guide before contributing to Taichi.

## General guidelines and tips

This section provides some general guidelines for the Taichi community and tips that we find practically useful.

### Be pragmatic & no overkills

Always use straightforward (sometimes even brute-force) solutions: Complicated code usually suggests a lack of design or over-engineering.

> - "There are two ways of constructing a software design: One way is to make it so simple that there are obviously no deficiencies, and the other way is to make it so complicated that there are no obvious deficiencies. *The first method is far more difficult*." — [C.A.R. Hoare](https://en.wikipedia.org/wiki/Tony_Hoare)
> - "Perfection (in design) is achieved not when there is nothing more to add, but rather when there is nothing more to take away." — [Antoine de Saint-Exupéry](https://en.wikipedia.org/wiki/The_Cathedral_and_the_Bazaar)

### Juxtapose pros and cons

When it comes to making a design decision, weigh up its pros and cons. A design is *good to go* so long as its advantages outweigh its disadvantages.

### Communicate effectively

Our ultimate goal is to build a sustainable, prosperous Taichi community, and effective communication is the cornerstone of that goal. Following are tips that may contribute to effective communication:

- Concise:
  - The message behind your words outweighs the number of your words. Use as few words as possible to drive your point home.
  - Use tables, figures, and lists where possible.

- Professional:
  - Read twice before you post: Would your point get across with your words?
  - Use a spell checker, such as [Grammarly](https://app.grammarly.com/), to improve your writing in terms of grammar, style, and tone.

- Constructive and courteous: Base your feedback and discussions on facts, *NOT* on personal feelings.
  - Acceptable😃: *"This design could be confusing to new Taichi users. If it were designed this way, it could..."*
  - Undesirable😞: ~~*"This design is terrible."*~~

## What you can contribute

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

## File an issue

If you would like to propose a new feature, or if you spot a potential issue, you can file an issue with Taichi.

:::note
When you try to report potential bugs in an issue, please consider running `ti diagnose` and offer its output as an attachment. This helps the maintainers to learn more about the context and the system information of your environment to make the debugging process more efficient and solve your issue more easily.
:::

:::caution
When filing your issue, review it once again to ensure that no sensitive information about your data or yourself creeps in.
:::

## Take over an issue

Except for minor updates, most PRs start from a developer taking over an issue. This section provides some corresponding tips and best practices.

### Where to find issues for starters

| Issue Tag                                                    | Description               | Target developer                               |
| ------------------------------------------------------------ | ------------------------- | ---------------------------------------------- |
| [good first issue](https://github.com/taichi-dev/taichi/issues?q=is:open+is:issue+label:"good+first+issue") | Issues that are easy to start with                          | Developers new to Taichi                       |
| [welcome contribution](https://github.com/taichi-dev/taichi/issues?q=is:open+is:issue+label:"welcome+contribution") | Issues *slightly* more challenging | Developers who wish to dive deeper into Taichi |

### Best practices

- When you plan to take over an issue:
  - **Best practice**: Leave a message claiming that you are working on it.
  - **Goal**: Avoid unnecessary repeated work.
  - **Example**: *"I know how to fix this and would like to help."*
- After you take over an issue:
  - **Best practice**:
      1. Briefly describe how you plan to handle it (if no solution has been provided).
      2. Hold off until a core developer responds to your action plan.
  - **Goal**: Keep your implementation neat and effective.
  - **Example**: See [#2610](https://github.com/taichi-dev/taichi/issues/2610).

## References for documentation updates

As part of the effort to increase visibility of the community and to improve developer experience, we highly recommend including documentation updates in your PR if applicable. Here are some of the documentation-specific references and tips:

- Documentation source files are hosted under [docs/](https://github.com/taichi-dev/taichi/blob/master/docs/).
- We use GitHub Flavored Markdown (GFM) and [Docusaurus](https://docusaurus.io/) to build our documentation site. For information on the supported Markdown syntax, see the  [Documentation Writing Guide](./doc_writing.md).
- When it comes to writing, we adhere to the [Google Developer Documentation Style Guide](https://developers.google.com/style/).
- For instructions on setting up a local server and previewing your updated documentation in real-time, see the [Local Development](https://github.com/taichi-dev/docs.taichi.graphics#local-development).

## Add test cases for your local changes

If your PR is to implement a new feature, we recommend that you write your own test cases to cover corner cases for your codes before filing a PR.

- To write a Python test case, see the [Workflow for writing a Python test](./write_test.md).
- To write a C++ test case, see the [Workflow for writing a C++ test](./writing_cpp_tests.md).

## Conduct style checks and integration tests locally

We highly recommend that you complete code style checks and integration tests on your local computer before filing a PR.

### Enforce code style

Taichi enforces code style via [pre-commit](https://pre-commit.com/) hooks, which includes the following checks:

1. C++ codes are formatted by `clang-format-10`.
2. Python codes are formatted by `yapf v0.31.0` based on PEP 8 rules.
3. Python codes are statically checked by [`pylint`](https://pylint.org/).

You will need to install `pre-commit` first:

```
pip install pre-commit
```

and run the code checkers:

```
pre-commit run -a
```

With this command, `yapf` will format your Python codes automatically.
You can install it as a pre-commit hook so that it is run before you commit the changes to git:

```
pre-commit install
```
<details>
<summary><font color="#006284"> What if I didn't format my code style locally? </font></summary>

No problem, the CI bot will run the code checkers and format your codes automatically when you submit a PR.

</details>


<!-- Todo: Make this a reusable fragment. -->

> For more style information for your C++ code, see [our C++ style](#c-style).

### Run integration tests

To run all the C++ and Python tests:
`python tests/run_tests.py`

- **Example 1:**
`python tests/run_tests.py -v -t3 -a cpu,metal -s`
  - `-v`: Verbose output.
  - `-t <threads>`: Set a custom number of threads for parallel testing.
  - `-a <arch(s)>`: Test only the specified backends (separated by comma).
  - `-s`: Original output from the tests.

- **Example 2:**
`python tests/run_tests.py numpy_io`
  - `<filename(s)>`: Run test cases in specified files only (separated by comma).
  - This command runs all tests in [tests/python/test_numpy_io.py](https://github.com/taichi-dev/taichi/blob/master/tests/python/test_numpy_io.py).

- **Example 3:**
`python tests/run_tests.py linalg -k "cross or diag"`
  - `-k <key>`: Run only the tests that match the specified keys (supports expression in a key string).
  - This command runs `test_cross()` and `test_diag()` in [tests/python/test_linalg.py](https://github.com/taichi-dev/taichi/blob/master/tests/python/test_linalg.py).

- **To show all available options**
`python tests/run_tests.py -h`

>  We have both Python and C++ test cases, but C++ test cases are disabled by default. To enable C++ test cases:
>
>  1. Build Taichi from source using the `python setup.py develop` command.
>  2. Set `TAICHI_CMAKE_ARGS="-DTI_BUILD_TESTS:BOOL=ON"`.

## File a pull request (PR)

Now you get to the point where you need to get your hands dirty with your PRs. This section provides the following:
- [Considerations when you create PRs](#considerations)
- [PR naming conventions](#pr-naming-conventions)
- [PR review & merging checklist](#pr-review-merging-checklist)

### Considerations

<!-- Todo: Consider turning this section into a mind map. -->

- **When implementing a complex feature:**

  - Consider breaking it down to multiple separate, self-contained PRs to provide the community with a clearer context and keep a more traceable development history.

- **When creating a PR:**

  - Have your PR address only one issue:
    - In this way, you keep your changesets small so that potential issues can be readily identified.
    - If you include in your PR irrevelant implementations, ensure that they are minor.
    - Your reviewers have the right to request you to remove massive, irrevelant changes from your PR.
  - If your PR is to implement a new feature, ensure that you have designed test cases for it. See [Add test cases for your local changes](#add-test-cases-for-your-local-changes).
  - You are required to conduct code style checks and integration tests locally for your PR. See [Conduct style checks and integration tests locally](#conduct-style-checks-and-integration-tests-locally)

- **When describing your PR:**
  - Provide sufficient information in the description of your PR to provide the community with clearer context:
    - Link to a specific GitHub issue if applicable, for example `fixes #<issue_number>`.
    - Share important design decisions in your description.

- **If you create a PR still in progress:**

  - Click **Convert to draft** on your PR page to convert the PR to draft, indicating that you are still working on it.
  - Click **Ready for review** when you are all set and up for a review.
  - See [Draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/) for more information.


### PR naming conventions

Your PR will make it into the commit history in the the master branch or even Taichi's release notes, therefore it is important to keep your PR title self-explanatory. This section describes our PR naming conventions:

```Gherkin
[tag1] [tag2]...[tagN] Your PR title must be short but carry necessary info

^----^ ^----^...^----^ ^--------------------------------------------------^
|      |        |      |
|      |        |      +---> Capitalize the initial of your title.
|      |        +---> Adjacent tags are separated with precisely one space.
|      +--->  Frequently used tags: [cuda], [lang], [ci], [ir], [refactor].
+--->  Prepend at least one tag to your PR title.
```

- **Tag naming conventions:**
  - Prepend at least one tag, such as `[lang]`, to your PR title.
  - If you have multiple tags, separate adjacent tags with one space.
  - See [misc/prtags.json](https://github.com/taichi-dev/taichi/blob/master/misc/prtags.json) for a full list of available tags.
  - We differentiate PRs for end-users from PRs for developers by *capitalizing tag initial*.
    - If a PR deals with a feature visible to the end-users, initialize the most relevant tag and the PR will [make it into the release notes](https://github.com/taichi-dev/taichi/blob/master/misc/make_changelog.py). For example, `[Metal]`, `[Vulkan]`, `[IR]`, `[Lang]`, or `[CUDA]`. Ensure that your PR title has *AT MOST* one tag dealt this way.
    - If a PR deals with the underlying or intermediate implementation, then it is for the developers and you need to ensure that all its tags are *in lowercase*. For example, `[metal]`, `[vulkan]`, `[ir]`, `[lang]`, or `[cuda]`.

  :::danger INCORRECT
  `[Lang][refactor]` (sans space)
  :::

  :::tip CORRECT
  `[Lang] [refactor]`
  :::

  :::danger INCORRECT
  `[GUI] [Mac] Support modifier keys` (both tags have their initial capitalized)
  :::

  :::tip CORRECT
  `[gui] [Mac] Support modifier keys` (only one tag has its initial capitalized)
  :::

- **Title naming conventions:**
  - Keep your PR title short enough but ensure that it carries necessary information.
  - Do not include back quotes ("\`") in your PR title.
  - Capitalize the initial letter of your title, which is the word immediately after your tag(s).

  :::danger INCORRECT
  `[Doc] improve documentation` (the initial of the title is not capitalized)
  :::

  :::tip CORRECT
  `[Doc] Improve documentation`
  :::

:::note

Following are some frequently used tags:

- `[cuda]`: Backend-specific changes.
- `[lang]`: Frontend language features, including syntax sugars.
- `[ir]`: Intermediate representation-specific changes.
- `[refactor]`: Code refactoring changes.
- `[ci]`: CI/CD workflow-specific changes.
- `[Doc]`: Documentation updates.

When introducing a new tag, ensure that you add it to [misc/prtags.json](https://github.com/taichi-dev/taichi/blob/master/misc/prtags.json) so that others can follow.

:::

### PR review & merging checklist

Follow this checklist during PR review or merging:

1. Ensure that your PR title follows our [naming conventions](#pr-naming-conventions).
2. Ensure that Taichi's master branch has a *linear history*. See [Linear vs Non-Linear History](https://idiv-biodiversity.github.io/git-knowledge-base/linear-vs-nonlinear.html) for more information.
3. Ensure that your PR passes all Continuous Integration (CI) tests before merging it.

   CI is triggered each time you push a commit to an open PR. It builds and tests all commits in your PR in multiple environments. Keep an eye on the CI test results:
   - A ✔️ on the left-hand side of a commit hash: CI has passed,
   - A ❌ on the left-hand side of a commit hash: CI has failed.

Here, we do not want to repeat some best practices summarized in the following Google blog articles. But please spare a couple of minutes reading them if your PR is being reviewed or if you are reviewing a PR. They have our recommendation!
  - [Code Health: Understanding Code In Review](https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html)
  - [Code Health: Respectful Reviews == Useful Reviews](https://testing.googleblog.com/2019/11/code-health-respectful-reviews-useful.html)
  - [How to have your PR merged quickly](https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html)

## C++ style

We generally follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). One major exception is the naming convention of functions: Taichi adopts the snake case for function naming, as opposed to the camel case [suggested in Google's style](https://google.github.io/styleguide/cppguide.html#Function_Names), e.g. `this_is_a_taichi_function()`.

Below we highlight some of the most widely used styles.

### Naming conventions

- Class and struct names should use the camel case, for example, `CodegenLlvm`.
  - Prefer capitalizing only the first letter of an acronym/abbreviation ([examples](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)).
- Variable names should use the snake case, for example, `llvm_context`.
- Private class member variable names should end with an `_`, for example, `id_to_snodes_`.
- Constant names should use the camel case, with a prefix `k`, for example, `constexpr int kTaichiMaxNumArgs = 64;`.
- Macros should start with `TI_`, for example, `TI_NOT_IMPLEMENTED`.
  - In general, avoid using macros as much as possible.
  - Avoid using `TI_NAMESPACE_BEGIN/END` in the new code.

### Rule of thumbs

- Use `const` as much as possible, for example, function parameter types, class member functions, and more.
- Provide default initializers to the class member variables, at least for the POD types.
  ```cpp
  class Foo {
   private:
    int x_{0};
    char* buf_{nullptr};
  };
  ```
- Embrace the smart pointers and avoid `new` and `delete`.
- Mark the constructor `explicit` to prevent the compiler from doing any implicit conversion.
- Avoid virtual function calls in the constructors or destructors ([explanation](https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP50-CPP.+Do+not+invoke+virtual+functions+from+constructors+or+destructors)).

## Deal with compilation warnings

Taichi implements warning-free code by turning on `-Werror` by default. This means that Taichi takes warnings as errors, and we highly recommend that you resolve a warning as soon as it occurs.

In the following section, we provide several practical tips for handling some of the common scenarios that you may encounter during CI compilation.

### Deal with warnings that occur when compiling third-party header files

There is little we can do to third-party warnings other than turning them off. To turn off or mute warnings from specific third-party header files, use the `SYSTEM` option when configuring `include_directories` in your CMake files. Then, the included header files are treated as system headers. See the following two examples taken from [cmake/TaichiCore.cmake](https://github.com/taichi-dev/taichi/blob/master/cmake/TaichiCore.cmake):
```
# Treat files under "external/Vulkan-Headers/include" as system headers and mute warnings from them.
include_directories(SYSTEM external/Vulkan-Headers/include)

# Treat files under "external/VulkanMemoryAllocator/include" as system headers for target "${CORE_LIBRARY_NAME}"
target_include_directories(${CORE_LIBRARY_NAME} SYSTEM PRIVATE external/VulkanMemoryAllocator/include)
```

### Deal with warnings when compiling third-party libraries or targets

Ideally, third-party libraries or targets ought to be built completely independent of your Taichi project. In practice, because of the design of the CMake system, CMake variables from the Taichi and third-party submodules are sometimes messed up. Therefore, we recommend that you disable warnings from a third-party library or target:

1. Separate the submodule's `CMAKE_CXX_FLAGS` from the same variable defined in Taichi.
2. Remove the `-Wall` option from the submodule's `CMAKE_CXX_FLAGS` variables.

### Mute specific warning types across the entire Taichi project
You can find details about how to mute certain warning types from the [Clang Compiler User Manual](https://clang.llvm.org/docs/UsersManual.html); it usually starts with `-Wno-`. Please explain what the warning is about and why we should ignore it in the comments.

The following examples can be found in [cmake/TaichiCXXFlags.cmake](https://github.com/taichi-dev/taichi/blob/master/cmake/TaichiCXXFlags.cmake):
```
# [Global] Clang warns if a C++ pointer's nullability was not explicitly marked (__nonnull, nullable, ...).
# Nullability seems to be a clang-specific feature; thus we disable this warning.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nullability-completeness ")

# [Global] By evaluating "constexpr", compiler throws a warning for functions known to be dead at compile time.
# However, some of these "constexpr" specifiers are debug flags and will be manually enabled upon debugging.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unneeded-internal-declaration ")
```

### Mute warnings from specific code blocks

:::caution
The approach presented here is *not* recommended and considered your last approach, because it is *not* reliable.
:::

In rare situations where you can neither fix nor mute the warnings from specific code blocks via conventional approaches, your last approach is to mute them by decorating your code block using the `#pragma clang diagnostic` macros. Beware that `#pragma`s are not defined in the C++ standard and that their implementations depend heavily on the compiler. That is to say, this solution is neither stable nor elegant.

To ignore all warnings from a specific code block, wrap it up with the following two groups of macros. Further, you can even replace `-Wall` with a group of warning types for finer control. See the following example:

```
#if defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wall"
#endif

{Your Code Goes Here}

#if defined(__clang__)
  #pragma clang diagnostic pop
#endif
```

## Handle special CI failures
Taichi's CI system is implemented using the [Github Actions](https://github.com/features/actions), the entrance of which lies in [testing.yaml](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/testing.yml). Depending on the CI pipeline, `testing.yml` will execute one of the corresponding test scripts under [this directory](https://github.com/taichi-dev/taichi/tree/master/.github/workflows/scripts)

There are a few CI pipelines that work slightly different from the standard CI pipeline:

### CI pipeline - Build Android Demos
`Build Andriod Demos` builds both [taichi-repo](https://github.com/taichi-dev/taichi) with your PR applied and an external [taichi-aot-demo](https://github.com/taichi-dev/taichi-aot-demo) repo. After that, it executes the demos from `taichi-aot-demo` with the just-compiled Taichi program and libraries.

If your PR to `taichi-repo` contains changes to some public interface, you may need to adjust the codes in `taichi-aot-demo` to avoid breaking the demos. To achieve that, please follow these steps:
1. File your PR to `taichi-repo`. If this PR changes the public interface, then it probably breaks the demos thus fail the `Build Android Demos` CI pipeline - Don't panic, this is expected.
2. Update the demo codes in `taichi-aot-demo` to make it work with the above mentioned PR, then file a separate PR to `taichi-aot-demo` repo and have it merged.
3. In the original PR to `taichi-repo`, update the commit id for `taichi-aot-demo` in [aot-demo.sh](https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/aot-demo.sh). This time your PR is expected to pass `Build Android Demos`.

##  Still have issues?

If you encounter any issue that is not covered here, feel free to ask us on GitHub discussions or [open an issue on GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md) with all the details attached. We are always there to help!

Finally, thanks again for your interest in contributing to Taichi. We look forward to seeing your contributions!
