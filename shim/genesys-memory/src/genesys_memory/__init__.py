"""Compatibility shim: ``genesys_memory`` was renamed to ``papez``.

Every ``genesys_memory.X`` import resolves to the *same* module object as
``papez.X`` (not a second copy), so classes, enums and singletons stay
identical whichever name a caller used.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

OLD, NEW = "genesys_memory", "papez"

warnings.warn(
    "genesys_memory has been renamed to papez. Replace `genesys_memory` with `papez` in your imports; "
    "this shim is not maintained.",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != OLD and not fullname.startswith(OLD + "."):
            return None
        real = NEW + fullname[len(OLD):]
        module = importlib.import_module(real)
        sys.modules[fullname] = module
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

_papez = importlib.import_module(NEW)
sys.modules[__name__] = _papez
