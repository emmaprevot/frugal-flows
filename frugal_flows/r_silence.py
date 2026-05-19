"""Silence R output forwarded to Python through rpy2.

R messages — including native-package compile/load spew (Eigen include paths,
linker flags, etc.) — reach Python via rpy2's ``writeconsole`` callbacks, not
through R's stdout/stderr connections. So ``ro.r('sink("/dev/null")')`` does
NOT suppress them. We have to replace the callbacks themselves.

Symptom this exists to fix: a single cell's output accumulates thousands of
compile-spew lines, JupyterLab's frontend tries to auto-link every one of them
by hitting /api/contents/..., and the resulting 404 storm makes the lab
unusable.
"""

from __future__ import annotations

from contextlib import contextmanager

from rpy2.rinterface_lib import callbacks


@contextmanager
def silence_r():
    """Suppress all R writeconsole output for the duration of the with-block.

    Usage::

        from frugal_flows.r_silence import silence_r

        with silence_r():
            data = causl_py.generate_mixed_samples(10000, CAUSAL_PARAMS, 3)
    """
    saved_regular = callbacks.consolewrite_print
    saved_warnerror = callbacks.consolewrite_warnerror
    callbacks.consolewrite_print = lambda s: None
    callbacks.consolewrite_warnerror = lambda s: None
    try:
        yield
    finally:
        callbacks.consolewrite_print = saved_regular
        callbacks.consolewrite_warnerror = saved_warnerror
