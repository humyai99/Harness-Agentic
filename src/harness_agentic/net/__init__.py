"""Network access for tools: the URL policy, the fetcher, and text extraction.

Separate from ``tools/`` on purpose. A tool that fetches a URL the model chose
is an SSRF primitive, and the defence against that belongs in one reviewable
place rather than at each call site -- the same reasoning that puts every
filesystem touch behind ``ExecEnvironment``.
"""
