"""Reading structured data: SQL sources, and a searchable local corpus.

Separate from ``tools/`` for the same reason ``net/`` is. The rules about what
an agent may read -- read-only statements, masked credential columns, bounded
result sets -- belong in one reviewable place, not at each call site.
"""
