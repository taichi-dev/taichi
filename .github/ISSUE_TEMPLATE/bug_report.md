---
name: Bug Report
about: Create a report to help us improve
title: ''
labels: potential bug
assignees: ''

---

<!-- We've collected some common issue solutions in https://docs.taichi-lang.org/docs/install. Make sure you've check them out first. Hopefully they could address your problem. -->

**Describe the bug**
A clear and concise description of what the bug is, ideally within 20 words.

**To Reproduce**
Please post a **minimal sample code** to reproduce the bug.
The developer team will put a higher priority on bugs that can be reproduced within 20 lines of code. If you want a prompt reply, please keep the sample code **short** and **representative**.

```py
# sample code here
```

**Log/Screenshots**
Please post the **full log** of the program (instead of just a few lines around the error message, unless the log is > 1000 lines). This will help us diagnose what's happening. For example:

```
$ python my_sample_code.py
[Taichi] mode=release
[Taichi] version 0.6.29, llvm 10.0.0, commit b63f6663, linux, python 3.8.3
...
```

**Additional comments**
If possible, please also consider attaching the output of command `ti diagnose`. This produces the detailed environment information and hopefully helps us diagnose faster.

If you have local commits (e.g. compile fixes before you reproduce the bug), please make sure you first make a PR to fix the build errors and then report the bug.
