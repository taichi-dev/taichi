---
sidebar_position: 6
---

# Command line utilities

A successful installation of Taichi should add a CLI (Command-Line
Interface) to your system, which is helpful to perform several rountine
tasks quickly. To invoke the CLI, please run `ti` or
`python3 -m taichi`.

## Examples

Taichi provides a set of bundled examples. You could run `ti example -h`
to print the help message and get a list of available example names.

For instance, to run the basic `fractal` example, try: `ti example fractal`
from your shell. (`ti example fractal.py` should also work)

You may print the source code of example by running
`ti example -p fractal`, or `ti example -P fractal` for print with
syntax highlight.

You may also save the example to current work directory by running
`ti example -s fractal`.

## Changelog

Sometimes it's convenient to view the changelog of the current version
of Taichi. To do so, you could run `ti changelog` in your shell.

## REPL Shell

Sometimes it's convenient to start a Python shell with
`import taichi as ti` as a pre-loaded module for fast testing and
confirmation. To do so from your shell, you could run `ti repl`.

## System information

When you try to report potential bugs in an issue, please consider
running `ti diagnose` and offer its output as an attachment. This could
help maintainers to learn more about the context and the system
information of your environment to make the debugging process more
efficient and solve your issue more easily.

:::caution
**Before posting it, please review and make sure there's no sensitive information about your data or yourself gets carried in.**
:::

## Converting PNGs to video

Sometimes it's convenient to convert a series of `png` files into a
single video when showing your result to others.

For example, suppose you have `000000.png`, `000001.png`, \... generated
according to [Export your results](./export_results.md) in the
**current working directory**.

Then you could run `ti video` to create a file `video.mp4` containing
all these images as frames (sorted by file name).

Use `ti video -f40` for creating a video with 40 FPS.

## Converting video to GIF

Sometimes we need `gif` images in order to post the result on forums.

To do so, you could run `ti gif -i video.mp4`, where `video.mp4` is the
`mp4` video (generated with instructions above).

Use `ti gif -i video.mp4 -f40` for creating a GIF with 40 FPS.
