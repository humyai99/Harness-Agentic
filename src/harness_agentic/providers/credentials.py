"""Finding credentials, and refusing to run when they are stored unsafely.

A world-readable ``.env`` is an incident, not a lint finding, so it stops
startup rather than printing a warning nobody reads. Resolution order puts the
process environment first so a one-off override never has to be written to
disk.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

from harness_agentic.constants import harness_home, profile_dir
from harness_agentic.errors import CredentialError, InsecureCredentialFile
from harness_agentic.providers.base import Credentials, CredentialSource
from harness_agentic.providers.catalog import ProviderInfo, provider_info

_QUOTED_MIN = 2
"""Shortest string that can be a matched pair of quotes."""


class Secret:
    """A credential that will not print itself.

    ``__repr__`` and ``__str__`` both redact, so an accidental f-string in a
    log line yields ``Secret(***)`` rather than an API key in a file that may
    be shipped to a log aggregator.
    """

    __slots__ = ("_value", "source")

    def __init__(self, value: str, *, source: str) -> None:
        """Wrap ``value``, recording where it came from."""
        self._value = value
        self.source = source

    def reveal(self) -> str:
        """Return the raw value. Call this only at the point of use."""
        return self._value

    def __repr__(self) -> str:
        """Redacted."""
        return f"Secret(***, source={self.source!r})"

    __str__ = __repr__

    def __bool__(self) -> bool:
        """Whether a value is present."""
        return bool(self._value)


class SecretResolver:
    """Looks up credentials across the environment and profile dotenv files."""

    def __init__(self, *, profile: str | None = None, strict_permissions: bool = True) -> None:
        """Build a resolver for one profile."""
        self._strict = strict_permissions
        self._files = [profile_dir(profile) / ".env", harness_home() / ".env"]
        self._cache: dict[str, Secret] = {}
        self._dotenv: dict[str, tuple[str, Path]] | None = None

    def get(self, name: str) -> Secret | None:
        """Resolve one credential by name."""
        if name in self._cache:
            return self._cache[name]

        value = os.environ.get(name)
        if value:
            secret = Secret(value, source="env")
            self._cache[name] = secret
            return secret

        for key, (found, path) in self._load_dotenv().items():
            if key == name and found:
                secret = Secret(found, source=f"dotenv:{path.name}")
                self._cache[name] = secret
                return secret
        return None

    def first(self, names: tuple[str, ...]) -> Secret | None:
        """Resolve the first of several accepted names."""
        for name in names:
            if secret := self.get(name):
                return secret
        return None

    def sources(self) -> Mapping[str, str]:
        """Which source each resolved credential came from. Never the values."""
        return {name: secret.source for name, secret in self._cache.items()}

    def _load_dotenv(self) -> dict[str, tuple[str, Path]]:
        if self._dotenv is not None:
            return self._dotenv
        merged: dict[str, tuple[str, Path]] = {}
        for path in self._files:
            if not path.exists():
                continue
            self._check_permissions(path)
            for key, value in _parse_dotenv(path).items():
                merged.setdefault(key, (value, path))
        self._dotenv = merged
        return merged

    def _check_permissions(self, path: Path) -> None:
        """Refuse to read a secrets file other users can read."""
        if not self._strict or os.name == "nt":
            return
        mode = path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            msg = (
                f"{path} is readable by other users (mode {stat.filemode(mode)}). "
                f"Run: chmod 600 {path}"
            )
            raise InsecureCredentialFile(msg)


def resolve_credentials(
    provider: str, resolver: SecretResolver, *, base_url_override: str | None = None
) -> Credentials:
    """Assemble credentials for one provider."""
    info: ProviderInfo | None = provider_info(provider)
    if info is None:
        msg = f"unknown provider {provider!r}"
        raise CredentialError(msg)

    base_url = base_url_override or info.base_url
    for env_name in info.base_url_env:
        if override := os.environ.get(env_name):
            base_url = override
            break

    secret = resolver.first(info.api_key_env) if info.api_key_env else None
    if info.requires_key and secret is None:
        wanted = " or ".join(info.api_key_env)
        msg = (
            f"no credential for {info.label}: set {wanted} in the environment "
            f"or in {harness_home() / '.env'} (chmod 600)"
        )
        raise CredentialError(msg)

    return Credentials(
        base_url=base_url,
        api_key=secret.reveal() if secret else None,
        source=_source_of(secret),
        extra_headers=info.quirks.extra_headers,
    )


def _source_of(secret: Secret | None) -> CredentialSource:
    """Map a secret's origin onto the Credentials source vocabulary."""
    if secret is None:
        return "none"
    if secret.source == "env":
        return "env"
    if secret.source.startswith("dotenv"):
        return "dotenv"
    return "config"


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a dotenv file.

    Intentionally small: no interpolation, no command substitution, no export
    semantics. A secrets file should be inert -- anything that can execute is a
    place for something unexpected to run at startup.
    """
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= _QUOTED_MIN and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values
