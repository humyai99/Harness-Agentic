"""Deciding which URLs an agent may fetch.

A tool that fetches a URL the *model* chose is a server-side request forgery
primitive, and it is pointed at the inside of whatever network the agent runs
on. The dangerous targets are not exotic:

* ``http://169.254.169.254/`` -- the cloud metadata service, which hands out
  IAM credentials to anything that asks from the instance.
* ``http://localhost:8080/`` -- whatever admin interface is bound to loopback
  and assumes only trusted callers can reach it.
* ``http://10.0.0.5/`` -- the internal service that has no authentication
  because it is "not exposed".

So the check is on the **resolved addresses**, not the hostname. Blocking a
denylist of names stops nothing: ``evil.test`` can resolve to ``127.0.0.1``,
and a name that looks external routinely points inward.

Residual risk, stated plainly: between validating an address and connecting to
it, DNS can change the answer -- classic rebinding. Pinning the connection to
the validated address closes that, and :func:`resolve_and_check` returns the
addresses so a transport can do exactly that. What this module guarantees on
its own is that the name resolved to something public at check time.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias
from urllib.parse import SplitResult, urlsplit, urlunsplit

IpAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver: TypeAlias = Callable[[str], Sequence[IpAddress]]
"""Turns a hostname into addresses. Injected so the policy is testable without
DNS, and so a deployment can supply its own."""

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 5
MAX_BYTES = 5_000_000
"""Enough for any page worth reading, small enough that a hostile endpoint
streaming forever cannot exhaust memory."""

BLOCKED_PORTS = frozenset({22, 23, 25, 465, 587, 3306, 5432, 6379, 9200, 11211, 27017})
"""Ports where an HTTP request is not a page fetch but a probe of, or an
injection into, something else."""

METADATA_ADDRESSES = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS, GCP, Azure, DigitalOcean
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDS over IPv6
    }
)


class UrlRefused(ValueError):
    """A URL was rejected before any connection was attempted."""


@dataclass(frozen=True, slots=True)
class Checked:
    """A URL that passed the policy, with what it resolved to."""

    url: str
    host: str
    port: int
    addresses: tuple[IpAddress, ...]

    def pinned(self) -> str:
        """The first validated address, for a transport that pins the connection."""
        return str(self.addresses[0])


@dataclass
class UrlPolicy:
    """What an agent is allowed to reach over the network."""

    allow_private: bool = False
    """Only ever true for a deliberately configured internal deployment, and
    then the operator has said so in writing."""
    allowed_hosts: frozenset[str] = frozenset()
    """When non-empty, nothing outside it is reachable. The strongest control
    available, and the right one for a support bot with three known APIs."""
    blocked_hosts: frozenset[str] = frozenset()
    max_redirects: int = MAX_REDIRECTS
    max_bytes: int = MAX_BYTES
    schemes: frozenset[str] = ALLOWED_SCHEMES
    blocked_ports: frozenset[int] = BLOCKED_PORTS
    resolver: Resolver = field(default_factory=lambda: system_resolver)

    def check(self, url: str) -> Checked:
        """Validate a URL, resolving it. Raises :class:`UrlRefused`."""
        parts = urlsplit(url.strip())
        if parts.scheme.lower() not in self.schemes:
            allowed = ", ".join(sorted(self.schemes))
            detail = f"{parts.scheme or 'that'} is not a fetchable scheme; only {allowed}"
            raise UrlRefused(detail)
        if parts.username or parts.password:
            # Credentials in a URL end up in logs and in the transcript.
            detail = "URLs with embedded credentials are refused; pass a header instead"
            raise UrlRefused(detail)

        host = (parts.hostname or "").lower()
        if not host:
            detail = f"{url!r} has no host"
            raise UrlRefused(detail)

        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
        if port in self.blocked_ports:
            detail = f"port {port} is not fetchable"
            raise UrlRefused(detail)

        if self.allowed_hosts and not _matches(host, self.allowed_hosts):
            allowed = ", ".join(sorted(self.allowed_hosts))
            detail = f"{host} is not on the allowlist ({allowed})"
            raise UrlRefused(detail)
        if _matches(host, self.blocked_hosts):
            detail = f"{host} is blocked"
            raise UrlRefused(detail)

        addresses = self.resolve_and_check(host)
        return Checked(url=_normalize(parts), host=host, port=port, addresses=addresses)

    def resolve_and_check(self, host: str) -> tuple[IpAddress, ...]:
        """Resolve a host and refuse it if *any* address is internal.

        Any, not all: a name that resolves to one public address and one
        loopback address is a rebinding attempt wearing a disguise, and picking
        the public one would walk straight into it.
        """
        # A literal address needs no resolution, and short-circuiting here
        # rather than inside the resolver means no injected resolver can let one
        # through by forgetting the case. `http://127.0.0.1/` is the first thing
        # anyone tries, and it still has to pass the checks below.
        addresses: Sequence[IpAddress]
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                addresses = self.resolver(host)
            except OSError as exc:
                detail = f"could not resolve {host}: {exc}"
                raise UrlRefused(detail) from exc
        if not addresses:
            detail = f"{host} resolved to nothing"
            raise UrlRefused(detail)

        for address in addresses:
            if address in METADATA_ADDRESSES:
                # Checked first so the refusal names the metadata service
                # rather than reporting it as merely "link-local", and refused
                # even when private addresses are permitted: there is no
                # legitimate reason for an agent to read the instance's own
                # credentials, and every reason for an attacker to want it to.
                detail = f"{host} resolves to the cloud metadata service at {address}"
                raise UrlRefused(detail)
            reason = internal_reason(address)
            if reason and not self.allow_private:
                detail = f"{host} resolves to {address}, which is {reason}"
                raise UrlRefused(detail)
        return tuple(addresses)


def internal_reason(address: IpAddress) -> str:  # noqa: PLR0911
    """Why an address is not safe to fetch from, or the empty string.

    One branch per reason so the refusal message names it. "10.0.0.5, which is
    a private network" tells an operator what to change; "blocked" does not.
    """
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        # Checked first, and delegated to the mapped address. `::ffff:127.0.0.1`
        # is loopback wearing an IPv6 hat, and which of the IPv6 flags happen to
        # be set for it varies by platform -- so asking the IPv4 address the
        # question is both more accurate and stable across them.
        return internal_reason(address.ipv4_mapped) or "an IPv4-mapped address"
    # Most specific reason first. Several of these overlap -- 0.0.0.0 is both
    # unspecified and private, fe80::1 is both link-local and private -- and the
    # narrower answer is the one an operator can act on.
    if address.is_loopback:
        return "the loopback interface"
    if address.is_unspecified:
        return "unspecified"
    if address.is_link_local:
        return "link-local"
    if address.is_multicast:
        return "multicast"
    if address.is_reserved:
        return "reserved"
    if address.is_private:
        return "a private network"
    return ""


def system_resolver(host: str) -> list[IpAddress]:
    """Resolve a hostname to every address the system knows for it."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    seen: dict[str, IpAddress] = {}
    for info in infos:
        raw = info[4][0]
        if isinstance(raw, str):
            seen[raw] = ipaddress.ip_address(raw)
    return list(seen.values())


def _matches(host: str, patterns: frozenset[str]) -> bool:
    """Whether a host matches a pattern set, honouring ``.suffix`` entries.

    ``example.test`` matches only itself; ``.example.test`` matches every
    subdomain. Without the distinction an allowlist entry for ``api.bank.test``
    would also permit ``api.bank.test.evil.test``.
    """
    for pattern in patterns:
        lowered = pattern.lower().lstrip("*")
        if lowered.startswith("."):
            if host == lowered[1:] or host.endswith(lowered):
                return True
        elif host == lowered:
            return True
    return False


def _normalize(parts: SplitResult) -> str:
    """Reassemble a URL without its fragment, which is never sent anyway."""
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path or "/", parts.query, ""))
