"""Run the doctests carried by the AOSS backend and the store selector.

These are the examples in the module docstrings. They are executed rather than
admired: a docstring that shows the wrong output is worse than none, because
the next reader believes it.
"""
from __future__ import annotations

import doctest
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maths_tools import store as store_module  # noqa: E402
from maths_tools.aoss import (  # noqa: E402
    client as client_module,
    config as config_module,
    mappings as mappings_module,
    signing as signing_module,
    store as aoss_store_module,
)

MODULES = [config_module, signing_module, mappings_module, client_module,
           aoss_store_module, store_module]

if __name__ == "__main__":
    passed = failed = 0
    detail = []
    for module in MODULES:
        # IGNORE_EXCEPTION_DETAIL: the examples name the exception class
        # without its module path, which is how a reader writes it.
        result = doctest.testmod(
            module, optionflags=doctest.IGNORE_EXCEPTION_DETAIL, verbose=False)
        passed += result.attempted - result.failed
        failed += result.failed
        if result.failed:
            detail.append((module.__name__, result.failed))

    print("PASS %d" % passed)
    for name, n in detail:
        print("FAIL %s\n     %d doctest(s) failed" % (name, n))
    print("FAILED %d" % failed)
    sys.exit(1 if failed else 0)
