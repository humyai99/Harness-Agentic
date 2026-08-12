"""Single source of truth for the package version.

Kept in its own module so ``hatch`` can read it without importing the package
(which would drag in dependencies at build time).
"""

__version__ = "0.0.1"
