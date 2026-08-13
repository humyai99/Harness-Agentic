"""The URL policy, the fetcher, and HTML extraction.

The policy tests are the ones that matter most in this file. A tool that fetches
a URL the model chose is an SSRF primitive pointed at the inside of the network
the agent runs on, and every one of these cases is a real bypass technique
rather than a hypothetical: a public name that resolves to loopback, a redirect
to the metadata service, an IPv4-mapped IPv6 address, a hostname that is a
prefix of an allowlisted one.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any

import httpx
import pytest

from harness_agentic.core.secrets import Secret
from harness_agentic.net.extract import extract, to_text
from harness_agentic.net.fetch import FetchResult, HttpFetcher, RecordedFetcher
from harness_agentic.net.policy import (
    METADATA_ADDRESSES,
    UrlPolicy,
    UrlRefused,
    internal_reason,
)
from harness_agentic.net.search import BraveSearch, SearchHit, StaticSearch, from_environment
from harness_agentic.providers.credentials import SecretResolver


def resolving(**mapping: str) -> Any:
    """A resolver that answers from a table, so no DNS is involved."""

    def resolve(host: str) -> list[Any]:
        # Literal addresses short-circuit, exactly as the real resolver does.
        # A fake that diverges here would let a test pass on a path production
        # never takes.
        try:
            return [ipaddress.ip_address(host)]
        except ValueError:
            pass
        if host not in mapping:
            detail = f"no such host {host}"
            raise OSError(detail)
        return [ipaddress.ip_address(part) for part in mapping[host].split(",")]

    return resolve


def public(**extra: str) -> UrlPolicy:
    return UrlPolicy(resolver=resolving(**{"example.test": "93.184.216.34", **extra}))


# -- the policy ------------------------------------------------------------------


def test_a_public_url_passes() -> None:
    checked = public().check("https://example.test/page?q=1#frag")
    assert checked.host == "example.test"
    assert checked.port == 443
    # The fragment is never sent, so it is not part of the checked URL.
    assert checked.url == "https://example.test/page?q=1"


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("127.0.0.1", "the loopback interface"),
        ("10.0.0.5", "a private network"),
        ("192.168.1.1", "a private network"),
        ("172.16.0.1", "a private network"),
        ("0.0.0.0", "unspecified"),  # noqa: S104 - the point is that it is refused
        ("::1", "the loopback interface"),
        ("fe80::1", "link-local"),
        # Loopback wearing an IPv6 hat. Which IPv6 flags are set for a mapped
        # address varies by platform, so the reason has to come from the mapped
        # address rather than from the flags.
        ("::ffff:127.0.0.1", "the loopback interface"),
    ],
)
def test_a_name_resolving_inward_is_refused(address: str, expected: str) -> None:
    # The check is on the resolved address, not the name. A denylist of names
    # stops nothing: anyone can point a public name at 127.0.0.1.
    policy = UrlPolicy(resolver=resolving(**{"innocent.test": address}))
    with pytest.raises(UrlRefused, match=re.escape(expected)):
        policy.check("http://innocent.test/")
    assert internal_reason(ipaddress.ip_address(address)) == expected


def test_the_metadata_address_is_refused_by_name() -> None:
    policy = UrlPolicy(resolver=resolving(**{"innocent.test": "169.254.169.254"}))
    with pytest.raises(UrlRefused, match="metadata service"):
        policy.check("http://innocent.test/")


def test_the_metadata_service_is_refused_even_when_private_is_allowed() -> None:
    # An internal deployment may legitimately need 10.0.0.0/8. It never needs
    # the instance's own IAM credentials.
    policy = UrlPolicy(allow_private=True, resolver=resolving(**{"meta.test": "169.254.169.254"}))
    with pytest.raises(UrlRefused, match="metadata"):
        policy.check("http://meta.test/latest/meta-data/")

    permitted = UrlPolicy(allow_private=True, resolver=resolving(**{"internal.test": "10.0.0.5"}))
    assert permitted.check("http://internal.test/").host == "internal.test"


def test_a_split_horizon_answer_is_refused_entirely() -> None:
    # One public address and one loopback address is a rebinding attempt in a
    # disguise; picking the public one would walk straight into it.
    policy = UrlPolicy(resolver=resolving(**{"mixed.test": "93.184.216.34,127.0.0.1"}))
    with pytest.raises(UrlRefused, match=re.escape("127.0.0.1")):
        policy.check("https://mixed.test/")


def test_only_http_schemes_are_fetchable() -> None:
    for url in ("file:///etc/passwd", "gopher://x.test/", "ftp://x.test/"):
        with pytest.raises(UrlRefused, match="scheme"):
            public().check(url)


def test_urls_with_embedded_credentials_are_refused() -> None:
    # They end up in logs and in the transcript.
    with pytest.raises(UrlRefused, match="credentials"):
        public().check("https://user:pass@example.test/")


def test_non_http_ports_are_refused() -> None:
    with pytest.raises(UrlRefused, match="port 6379"):
        public().check("http://example.test:6379/")


def test_an_allowlist_is_not_a_prefix_match() -> None:
    # Without this, an entry for api.bank.test also permits
    # api.bank.test.evil.test, which is a domain an attacker can register.
    policy = UrlPolicy(
        allowed_hosts=frozenset({"api.bank.test"}),
        resolver=resolving(
            **{
                "api.bank.test": "93.184.216.34",
                "api.bank.test.evil.test": "93.184.216.35",
            }
        ),
    )
    assert policy.check("https://api.bank.test/v1").host == "api.bank.test"
    with pytest.raises(UrlRefused, match="allowlist"):
        policy.check("https://api.bank.test.evil.test/v1")


def test_a_leading_dot_allows_subdomains() -> None:
    policy = UrlPolicy(
        allowed_hosts=frozenset({".bank.test"}),
        resolver=resolving(**{"api.bank.test": "93.184.216.34", "bank.test": "93.184.216.34"}),
    )
    assert policy.check("https://api.bank.test/").host == "api.bank.test"
    assert policy.check("https://bank.test/").host == "bank.test"


def test_a_blocklist_refuses_by_name() -> None:
    policy = UrlPolicy(
        blocked_hosts=frozenset({"bad.test"}), resolver=resolving(**{"bad.test": "93.184.216.34"})
    )
    with pytest.raises(UrlRefused, match="blocked"):
        policy.check("https://bad.test/")


def test_an_unresolvable_host_is_refused_not_crashed() -> None:
    with pytest.raises(UrlRefused, match="could not resolve"):
        public().check("https://nowhere.test/")


def test_a_literal_ip_is_checked_without_dns() -> None:
    # An agent that writes out the IP directly must hit the same wall.
    with pytest.raises(UrlRefused, match="loopback"):
        UrlPolicy().check("http://127.0.0.1:8080/admin")


def test_every_metadata_address_is_recognised() -> None:
    assert all(internal_reason(a) or a in METADATA_ADDRESSES for a in METADATA_ADDRESSES)


# -- the fetcher -----------------------------------------------------------------


def fetcher(handler: Any, **policy_kwargs: Any) -> HttpFetcher:
    return HttpFetcher(
        policy=public(**policy_kwargs.pop("hosts", {})),
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False),
    )


def test_a_body_comes_back_decoded() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="สวัสดี", headers={"content-type": "text/html"})

    result = fetcher(handler).get("https://example.test/")
    assert result.ok
    assert result.is_text
    assert result.text() == "สวัสดี"


def test_every_redirect_hop_is_re_checked() -> None:
    # This is the standard SSRF filter bypass: a public first hop that 302s
    # somewhere internal.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.test":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/creds"})
        return httpx.Response(200, text="secrets")

    client = fetcher(handler)
    with pytest.raises(UrlRefused, match="metadata"):
        client.get("https://example.test/redirect")


def test_a_permitted_redirect_is_followed_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.test/end"})
        return httpx.Response(200, text="arrived", headers={"content-type": "text/plain"})

    result = fetcher(handler).get("https://example.test/start")
    assert result.text() == "arrived"
    assert result.redirects == ("https://example.test/end",)


def test_authorization_is_dropped_when_a_redirect_changes_host() -> None:
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("authorization")))
        if request.url.host == "example.test":
            return httpx.Response(302, headers={"location": "https://other.test/x"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    client = HttpFetcher(
        policy=public(**{"other.test": "93.184.216.99"}),
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False),
    )
    client.get("https://example.test/", headers={"Authorization": "Bearer secret"})

    assert seen[0] == ("example.test", "Bearer secret")
    # The credential was meant for the first host, not whoever it points at.
    assert seen[1] == ("other.test", None)


def test_a_redirect_loop_is_refused() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.test/again"})

    with pytest.raises(UrlRefused, match="redirects"):
        fetcher(handler).get("https://example.test/loop")


def test_the_body_is_capped_while_streaming() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 5000, headers={"content-type": "text/plain"})

    client = HttpFetcher(
        policy=UrlPolicy(max_bytes=100, resolver=resolving(**{"example.test": "93.184.216.34"})),
        client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False),
    )
    result = client.get("https://example.test/big")
    assert len(result.body) == 100
    assert result.truncated


def test_a_binary_response_is_recognised_as_unreadable() -> None:
    result = FetchResult(
        url="https://x.test/a.png", status=200, content_type="image/png", body=b"\x89PNG"
    )
    assert not result.is_text


def test_the_recorded_fetcher_still_enforces_the_policy() -> None:
    # A fake that skips the policy hides the bugs the policy exists to catch.
    fake = RecordedFetcher(policy=public())
    fake.add("https://example.test/", "<p>hello</p>")
    assert "hello" in fake.get("https://example.test/").text()
    with pytest.raises(UrlRefused):
        fake.get("http://127.0.0.1/")


# -- extraction ------------------------------------------------------------------

PAGE = """
<html><head><title>The Title</title>
<style>body{color:red}</style>
<script>alert('x')</script></head>
<body>
<nav><a href="/nav">navigation</a></nav>
<h1>Heading</h1>
<p>First paragraph with <strong>emphasis</strong>.</p>
<ul><li>one</li><li>two</li></ul>
<p>See <a href="/docs">the docs</a> and <a href="https://out.test/x">outside</a>.</p>
<pre>def f():
    return 1</pre>
<footer>copyright</footer>
</body></html>
"""


def test_extraction_keeps_prose_and_drops_machinery() -> None:
    page = extract(PAGE, base_url="https://example.test/article")
    assert page.title == "The Title"
    assert "First paragraph with emphasis." in page.text
    assert "- one" in page.text
    # Script and style contents are never prose.
    assert "alert" not in page.text
    assert "color:red" not in page.text
    # Navigation and footers are chrome, and cost tokens on every page.
    assert "navigation" not in page.text
    assert "copyright" not in page.text


def test_extraction_resolves_links_against_the_page() -> None:
    page = extract(PAGE, base_url="https://example.test/article")
    hrefs = dict(page.links)
    assert hrefs["the docs"] == "https://example.test/docs"
    assert hrefs["outside"] == "https://out.test/x"
    assert "/nav" not in hrefs.values()


def test_extraction_preserves_code() -> None:
    page = extract(PAGE)
    assert "def f():\n    return 1" in page.text


def test_render_says_how_much_it_cut() -> None:
    # A silently truncated page makes the model answer confidently from the
    # first third of an article.
    page = extract(f"<html><body><p>{'word ' * 2000}</p></body></html>")
    rendered = page.render(max_chars=200)
    assert "truncated" in rendered
    assert len(rendered) < 600


def test_malformed_markup_still_yields_what_parsed() -> None:
    page = extract("<p>before<div><span>after</p></div>")
    assert "before" in page.text
    assert "after" in page.text


def test_non_html_is_only_tidied() -> None:
    assert to_text('{"a": 1}', content_type="application/json") == '{"a": 1}'


def test_invisible_characters_are_collapsed() -> None:
    # No-break and zero-width spaces are everywhere in real HTML and reach the
    # model as invisible token noise.
    page = extract("<p>a\u00a0b\u200bc</p>")
    assert page.text == "a b c"


# -- search ----------------------------------------------------------------------


def test_a_static_provider_answers_from_its_table() -> None:
    provider = StaticSearch(hits={"thai food": [SearchHit("Som Tam", "https://x.test/1", "salad")]})
    assert provider.search("thai food")[0].title == "Som Tam"
    assert provider.search("nothing") == []
    assert provider.queries == ["thai food", "nothing"]


def test_brave_results_are_parsed_and_snippets_stripped() -> None:
    fake = RecordedFetcher(policy=public(**{"api.search.brave.com": "93.184.216.50"}))
    fake.responses["https://api.search.brave.com/res/v1/web/search?q=curry&count=2"] = FetchResult(
        url="https://api.search.brave.com/",
        status=200,
        content_type="application/json",
        body=b'{"web":{"results":[{"title":"Curry","url":"https://x.test/c",'
        b'"description":"a <strong>spicy</strong> dish"}]}}',
    )
    provider = BraveSearch(api_key=Secret("k", source="test"), fetcher=fake)
    hits = provider.search("curry", limit=2)

    assert hits[0].url == "https://x.test/c"
    assert hits[0].snippet == "a spicy dish"
    # The key travels in a header, never in the query string.
    assert "X-Subscription-Token" in fake.sent[0][2]
    assert "k" not in fake.sent[0][1]


def test_a_malformed_provider_payload_yields_no_hits_rather_than_crashing() -> None:
    fake = RecordedFetcher(policy=public(**{"api.search.brave.com": "93.184.216.50"}))
    fake.responses["https://api.search.brave.com/res/v1/web/search?q=x&count=5"] = FetchResult(
        url="https://api.search.brave.com/",
        status=200,
        content_type="application/json",
        body=b"not json",
    )
    provider = BraveSearch(api_key=Secret("k", source="test"), fetcher=fake)
    assert provider.search("x") == []


def test_no_key_means_no_search_provider_at_all() -> None:
    # None rather than a stub: the tool is omitted, so the model never learns
    # that asking for a search sometimes produces an apology.
    assert from_environment(SecretResolver(), RecordedFetcher()) is None


def test_a_key_selects_its_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    provider = from_environment(SecretResolver(), RecordedFetcher())
    assert provider is not None
    assert provider.name == "tavily"
