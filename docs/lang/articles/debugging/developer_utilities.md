---
sidebar_position: 2
---

# Developer Utilities

This section provides a detailed description of some commonly used
utilities for Taichi developers.

## Logging

Taichi uses [spdlog](https://github.com/gabime/spdlog) as its logging
system. Logs can have different levels, from low to high, they are:

| LEVELS |
| ------ |
| trace  |
| debug  |
| info   |
| warn   |
| error  |

The higher the level is, the more critical the message is.

The default logging level is `info`. You may override the default
logging level by:

1.  Setting the environment variable like `export TI_LOG_LEVEL=warn`.
2.  Setting the log level from Python side:
    `ti.set_logging_level(ti.WARN)`.

In **Python**, you may write logs using the `ti.*` interface:

```python
# Python
ti.trace("Hello world!")
ti.debug("Hello world!")
ti.info("Hello world!")
ti.warn("Hello world!")
ti.error("Hello world!")
```

In **C++**, you may write logs using the `TI_*` interface:

```cpp
// C++
TI_TRACE("Hello world!");
TI_DEBUG("Hello world!");
TI_INFO("Hello world!");
TI_WARN("Hello world!");
TI_ERROR("Hello world!");
```

If one raises a message of the level `error`, Taichi will be
**terminated** immediately and result in a `RuntimeError` on Python
side.

```cpp
// C++
int func(void *p) {
  if (p == nullptr)
    TI_ERROR("The pointer cannot be null!");

  // will not reach here if p == nullptr
  do_something(p);
}
```

:::note
For people from Linux kernels, `TI_ERROR` is just `panic`.
:::

You may also simplify the above code by using `TI_ASSERT`:

```cpp
int func(void *p) {
  TI_ASSERT_INFO(p != nullptr, "The pointer cannot be null!");
  // or
  // TI_ASSERT(p != nullptr);

  // will not reach here if p == nullptr
  do_something(p);
}
```

## Debug taichi program using `gdb`

1. Prepare a script that can reproduce the issue, e.g. `python repro.py`.
2. Build taichi with debug information using `DEBUG=1 python setup.py develop` (or `install`).
3. Run `gdb --args python repro.py`, now you can debug from there! For example, you can set a
   breakpoint using `b foo.cpp:102` or `b Program::compile()`.

However if your issue cannot be reproduced consistently this solution isn't a great fit.
In that case it's recommended to follow the section below so that gdb is triggered automatically
when the program crashes.

## (Linux only) Trigger `gdb` when programs crash

```python
# Python
ti.init(gdb_trigger=True)
```

```cpp
// C++
CoreState::set_trigger_gdb_when_crash(true);
```

```bash
# Shell
export TI_GDB_TRIGGER=1
```

:::note
**Quickly pinpointing segmentation faults/assertion failures using**
`gdb`: When Taichi crashes, `gdb` will be triggered and attach to the
current thread. You might be prompt to enter sudo password required for
gdb thread attaching. After entering `gdb`, check the stack backtrace
with command `bt` (`backtrace`), then find the line of code triggering
the error.
:::
