---
sidebar_position: 10
---

# Versioning and releases

## Pre-1.0 versioning

Taichi follows [Semantic Versioning 2.0.0](https://semver.org/).

Since Taichi is still under version 1.0.0, we use minor version bumps
(e.g., `0.6.17->0.7.0`) for breaking API changes, and patch version
bumps (e.g., `0.6.9->0.6.10`) for backward-compatible changes.

## Workflow: releasing a new version

- Trigger a Linux build on
  [Jenkins](http://f11.csail.mit.edu:8080/job/taichi/) to see if
  CUDA passes all tests. Note that Jenkins is the only build bot we
  have that tests CUDA. (This may take half an hour.)

- Create a branch for the release PR, forking from the latest commit
  of the `master` branch.

  - Update Taichi version number at the beginning of
    `CMakeLists.txt`. For example, change
    `SET(TI_VERSION_PATCH 9)` to `SET(TI_VERSION_PATCH 10)` for
    a patch release.
  - commit with message "[release] vX.Y.Z", e.g.
    "[release] v0.6.10".
  - You should see two changes in this commit: one line in
    `CMakeLists.txt` and one line in `docs/version`.
  - Execute `ti changelog` and save its outputs. You will need
    this later.

- Open a PR titled "[release] vX.Y.Z" with the branch and commit
  you just now created.

  - Use the `ti changelog` output you saved in the previous step
    as the content of the PR description.
  - Wait for all the checks and build bots to complete. (This step
    may take up to two hours).

- Squash and merge the PR.

- Trigger the Linux build on Jenkins, again, so that Linux packages
  are uploaded to PyPI.

- Wait for all build bots to finish. This step uploads PyPI packages
  for macOS and Windows. You may have to wait for up to two hours.

- Update the `stable` branch so that the head of that branch is your
  release commit on `master`.

- Draft a new release
  [(here)](https://github.com/taichi-dev/taichi/releases):

  - The title should be \"vX.Y.Z\".
  - The tag should be \"vX.Y.Z\".
  - Target should be \"recent commit\" -\> the release commit.
  - The release description should be copy-pasted from the release
    PR description.
  - Click the \"Publish release\" button.

## Release cycle

Taichi releases new versions twice per week:

- The first release happens on Wednesdays.
- The second release happens on Saturdays.

Additional releases may happen if anything needs an urgent fix.
