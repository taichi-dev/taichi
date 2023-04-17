# -*- coding: utf-8 -*-

# -- stdlib --
import importlib
import os
import pathlib
import sys
import tempfile
from collections import namedtuple

# -- third party --
import cffi
import pytest

# -- own --

# -- code --

# Modified from pytest-timeout
tle_cffi = cffi.FFI()
tle_cffi.cdef(
    """
    void init(void);
    void set(int seconds, char *message);
    void cancel(void);
"""
)

TLE_WIN32 = r"""
#include <string.h>
#include <windows.h>

static HANDLE hWatchThread = NULL;
static HANDLE hRawStdErr = NULL;
static char *current_message;
static ULONGLONG deadline = (ULONGLONG)-1;


static DWORD WINAPI WatchThread(LPVOID lpParam) {
    while(1) {
        Sleep(100);
        if(GetTickCount64() < deadline) continue;
        HANDLE hStdErr = GetStdHandle(STD_ERROR_HANDLE);
        WriteFile(hRawStdErr, "\n\n===== TimeLimitExceeded: ", 27, NULL, NULL);
        int len = lstrlen(current_message);
        WriteFile(hRawStdErr, current_message, len, NULL, NULL);
        WriteFile(hRawStdErr, " =====\n\n", 8, NULL, NULL);
        ExitProcess(1);
    }
}

static void init(void) {
    HANDLE hStdErr = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE hProc = GetCurrentProcess();
    DuplicateHandle(
        hProc, hStdErr,
        hProc, &hRawStdErr,
        DUPLICATE_SAME_ACCESS,
        FALSE, 0
    );
}

static void set(int seconds, char *message) {
    if(hWatchThread == NULL) {
        hWatchThread = CreateThread(
            NULL, 0, WatchThread, 0, 0, NULL
        );
    }
    deadline = GetTickCount64() + seconds * 1000;
    current_message = strdup(message);
}

static void cancel(void) {
    deadline = (ULONGLONG)-1;
    if(current_message != NULL) {
        free(current_message);
    }
    current_message = NULL;
}
"""

TLE_POSIX = r"""
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <signal.h>

static char *current_message = NULL;
static int fd_stderr = 0;

static void enforce_limit(int sig) {
    dprintf(fd_stderr, "\n\n===== TimeLimitExceeded: %s =====\n\n", current_message);
    fflush(stdout);
    _exit(1);
}

static void init(void) {
    fd_stderr = dup(2);
}

static void set(int seconds, char *test) {
    current_message = strdup(test);
    signal(SIGALRM, enforce_limit);
    alarm(seconds);
}

static void cancel(void) {
    if(current_message != NULL) {
        free(current_message);
        current_message = NULL;
    }
    alarm(0);
    signal(SIGALRM, SIG_DFL);
}
"""

if sys.platform == "win32":
    tle_cffi.set_source("_hard_tle", TLE_WIN32)
else:
    tle_cffi.set_source("_hard_tle", TLE_POSIX)

# libdir = pathlib.Path(tempfile.gettempdir()) / 'hard_tle'
# libdir.mkdir(parents=True, exist_ok=True)
libdir = pathlib.Path(tempfile.mkdtemp())
tle_cffi.compile(tmpdir=str(libdir))
sys.path.append(str(libdir))
tle = importlib.import_module("_hard_tle")

tle.lib.init()

TIMEOUT_DESC = """
Timeout in seconds before dumping the stacks.  Default is 0 which
means no timeout.
""".strip()
FUNC_ONLY_DESC = """
When set to True, defers the timeout evaluation to only the test
function body, ignoring the time it takes when evaluating any fixtures
used in the test.
""".strip()

Settings = namedtuple("Settings", ["timeout", "func_only"])


@pytest.hookimpl
def pytest_addoption(parser):
    """Add options to control the timeout plugin."""
    group = parser.getgroup(
        "timeout",
        "Interrupt test run and dump stacks of all threads after a test times out",
    )
    group.addoption("--timeout", type=float, help=TIMEOUT_DESC)
    parser.addini("timeout", TIMEOUT_DESC)
    parser.addini("timeout_func_only", FUNC_ONLY_DESC, type="bool")


class TimeoutHooks:
    """Timeout specific hooks."""

    @pytest.hookspec(firstresult=True)
    def pytest_timeout_set_timer(item, settings):
        """Called at timeout setup.

        'item' is a pytest node to setup timeout for.

        Can be overridden by plugins for alternative timeout implementation strategies.

        """

    @pytest.hookspec(firstresult=True)
    def pytest_timeout_cancel_timer(item):
        """Called at timeout teardown.

        'item' is a pytest node which was used for timeout setup.

        Can be overridden by plugins for alternative timeout implementation strategies.

        """


def pytest_addhooks(pluginmanager):
    """Register timeout-specific hooks."""
    pluginmanager.add_hookspecs(TimeoutHooks)


@pytest.hookimpl
def pytest_configure(config):
    """Register the marker so it shows up in --markers output."""
    config.addinivalue_line(
        "markers",
        "timeout(timeout, func_only=False): Set a timeout "
        "and func_only evaluation on just one test item.  The first "
        "argument, *timeout*, is the timeout in seconds."
        "The func_only* keyword, when set to True, defers the timeout evaluation "
        "to only the test function body, ignoring the time it takes when "
        "evaluating any fixtures used in the test.",
    )

    settings = get_env_settings(config)
    config._env_timeout = settings.timeout
    config._env_timeout_func_only = settings.func_only


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item):
    """Hook in timeouts to the runtest protocol.

    If the timeout is set on the entire test, including setup and
    teardown, then this hook installs the timeout.  Otherwise
    pytest_runtest_call is used.
    """
    hooks = item.config.pluginmanager.hook
    settings = _get_item_settings(item)
    is_timeout = settings.timeout is not None and settings.timeout > 0
    if is_timeout and settings.func_only is False:
        hooks.pytest_timeout_set_timer(item=item, settings=settings)
    yield
    if is_timeout and settings.func_only is False:
        hooks.pytest_timeout_cancel_timer(item=item)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Hook in timeouts to the test function call only.

    If the timeout is set on only the test function this hook installs
    the timeout, otherwise pytest_runtest_protocol is used.
    """
    hooks = item.config.pluginmanager.hook
    settings = _get_item_settings(item)
    is_timeout = settings.timeout is not None and settings.timeout > 0
    if is_timeout and settings.func_only is True:
        hooks.pytest_timeout_set_timer(item=item, settings=settings)
    yield
    if is_timeout and settings.func_only is True:
        hooks.pytest_timeout_cancel_timer(item=item)


@pytest.hookimpl(tryfirst=True)
def pytest_report_header(config):
    """Add timeout config to pytest header."""
    if config._env_timeout:
        return [
            "timeout: %ss\ntimeout func_only: %s"
            % (
                config._env_timeout,
                config._env_timeout_func_only,
            )
        ]


@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(node):
    """Stop the timeout when pytest enters pdb in post-mortem mode."""
    hooks = node.config.pluginmanager.hook
    hooks.pytest_timeout_cancel_timer(item=node)


@pytest.hookimpl(trylast=True)
def pytest_timeout_set_timer(item, settings):
    """Setup up a timeout trigger and handler."""
    tle.lib.set(int(settings.timeout), str(item).encode("utf-8"))
    return True


@pytest.hookimpl(trylast=True)
def pytest_timeout_cancel_timer(item):
    """Cancel the timeout trigger if it was set."""
    tle.lib.cancel()
    return True


def get_env_settings(config):
    """Return the configured timeout settings.

    This looks up the settings in the environment and config file.
    """
    timeout = config.getvalue("timeout")
    timeout = timeout or os.environ.get("PYTEST_TIMEOUT", None)
    timeout = timeout or config.getini("timeout")
    timeout = int(timeout)

    func_only = config.getini("timeout_func_only")
    if func_only == []:
        func_only = None
    if func_only is not None:
        func_only = bool(func_only)
    return Settings(timeout, func_only or False)


def _get_item_settings(item, marker=None):
    """Return (timeout, method) for an item."""
    timeout = func_only = None
    if not marker:
        marker = item.get_closest_marker("timeout")
    if marker is not None:
        settings = _parse_marker(item.get_closest_marker(name="timeout"))
        timeout = settings.timeout
        func_only = bool(settings.func_only)
    if timeout is None:
        timeout = item.config._env_timeout
    if func_only is None:
        func_only = item.config._env_timeout_func_only
    if func_only is None:
        func_only = False
    return Settings(timeout, func_only)


def _parse_marker(marker):
    """Return timeout from marker.

    Either could be None.  The values are not interpreted, so
    could still be bogus and even the wrong type.
    """

    def get_settings(timeout=None, method=None):
        if (timeout, method) == (None, None):
            raise TypeError("Timeout marker must have at least one argument")
        return Settings(timeout, method)

    return get_settings(*marker.args, **marker.kwargs)
