"""The evidence store: one SQLite file, the only writer of every record (plan 3.4).

``from waterwhite.store import Store`` is the whole public surface of the package.
The result types of the read methods live in ``waterwhite.store.views``.
"""

from .store import Store

__all__ = ["Store"]
