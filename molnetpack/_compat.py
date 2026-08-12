"""Internal helpers for keeping deprecated names importable."""

import functools
import warnings


def deprecated_alias(func, old_name):
    """Return a wrapper that forwards to ``func`` but warns when called under ``old_name``."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name}() is deprecated; use {func.__name__}() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    return wrapper
