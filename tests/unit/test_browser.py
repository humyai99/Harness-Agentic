"""The page model, and the browser tools over a scripted driver.

No Chromium is downloaded. What is being tested is the part that is actually
tricky and would still be tricky with a real browser: what the outline contains,
what an element reference means, and what happens when one goes stale.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any

import pytest

from harness_agentic.browser.driver import BrowserError, FakeDriver
from harness_agentic.browser.page import Builder, RefTable, StaleReference
from harness_agentic.net.policy import UrlPolicy, UrlRefused
from harness_agentic.tools.approval import ApprovalPolicy, Mode, always_deny
from harness_agentic.tools.builtin.browser import install_browser_tools
from harness_agentic.tools.registry import ToolRegistry
from harness_agentic.tools.spec import Danger

LOGIN_PAGE: dict[str, Any] = {
    "role": "WebArea",
    "name": "Sign in",
    "children": [
        {"role": "generic", "children": [{"role": "heading", "name": "Sign in to Example"}]},
        {"role": "textbox", "name": "Email"},
        {"role": "textbox", "name": "Password"},
        {"role": "checkbox", "name": "Remember me", "checked": False},
        {"role": "button", "name": "Sign in"},
        {"role": "button", "name": "Delete account", "disabled": True},
        {"role": "link", "name": "Forgot your password?"},
        {"role": "paragraph", "name": "By signing in you agree to the terms."},
    ],
}

HOME_PAGE: dict[str, Any] = {
    "role": "WebArea",
    "name": "Dashboard",
    "children": [
        {"role": "heading", "name": "Welcome back"},
        {"role": "button", "name": "Log out"},
    ],
}


def resolver(host: str) -> list[Any]:
    del host
    return [ipaddress.ip_address("93.184.216.34")]


def driver() -> FakeDriver:
    fake = FakeDriver(policy=UrlPolicy(resolver=resolver))
    fake.add("https://example.test/login", LOGIN_PAGE, title="Sign in")
    fake.add("https://example.test/home", HOME_PAGE, title="Dashboard")
    fake.pages["https://example.test/login"]["links"] = {"e4": "https://example.test/home"}
    fake.start()
    return fake


# -- the page model ---------------------------------------------------------------


def test_only_interactive_elements_get_references() -> None:
    # A reference on a paragraph is noise: there is nothing to do with it, and it
    # doubles the size of the outline.
    snapshot = Builder().build(LOGIN_PAGE, url="https://example.test/login", title="Sign in")
    referenced = {node.role for node in snapshot.root.walk() if node.ref}

    assert referenced == {"textbox", "checkbox", "button", "link"}
    assert not any(node.ref for node in snapshot.root.walk() if node.role == "paragraph")


def test_a_disabled_element_gets_no_reference_and_says_so() -> None:
    snapshot = Builder().build(LOGIN_PAGE, url="https://x.test", title="t")
    delete = next(n for n in snapshot.root.walk() if n.name == "Delete account")

    assert delete.disabled
    assert not delete.ref
    assert "disabled" in "\n".join(snapshot.root.render())


def test_wrapper_roles_collapse_without_losing_their_children() -> None:
    # Keeping generic wrappers doubles the outline and tells the model nothing.
    snapshot = Builder().build(LOGIN_PAGE, url="https://x.test", title="t")
    rendered = snapshot.render()

    assert "generic" not in rendered
    assert "Sign in to Example" in rendered


def test_the_outline_is_readable_and_actionable() -> None:
    snapshot = Builder().build(LOGIN_PAGE, url="https://example.test/login", title="Sign in")
    rendered = snapshot.render()

    assert "https://example.test/login" in rendered
    assert 'textbox "Email" [ref=e1]' in rendered
    assert 'checkbox "Remember me" unchecked [ref=e3]' in rendered


def test_references_are_reminted_on_every_snapshot() -> None:
    # A reference is a promise about a page that existed a moment ago.
    builder = Builder()
    first = builder.build(LOGIN_PAGE, url="https://x.test", title="t")
    second = builder.build(LOGIN_PAGE, url="https://x.test", title="t")

    assert second.generation == first.generation + 1


def test_a_stale_reference_is_refused_with_a_way_forward() -> None:
    # A click on a stale reference lands on whatever moved into that position,
    # and doing it silently is how an agent cancels an order it meant to confirm.
    table = RefTable()
    table.replace(1, {"e1": "handle-one"})
    assert table.resolve("e1") == "handle-one"

    table.replace(2, {"e9": "handle-nine"})
    with pytest.raises(StaleReference) as caught:
        table.resolve("e1")
    assert "fresh snapshot" in str(caught.value)
    assert "e9" in str(caught.value)


def test_find_narrows_a_page_without_re_reading_it() -> None:
    snapshot = Builder().build(LOGIN_PAGE, url="https://x.test", title="t")
    matches = snapshot.find("sign in")

    assert [node.name for node in matches] == ["Sign in"]
    assert snapshot.find("nothing here") == []
    # Disabled elements are not offered as things to click.
    assert all(not node.disabled for node in snapshot.find("account"))


def test_a_huge_outline_says_how_much_it_cut() -> None:
    wide = {
        "role": "WebArea",
        "name": "big",
        "children": [{"role": "button", "name": f"Button {i}"} for i in range(400)],
    }
    rendered = Builder().build(wide, url="https://x.test", title="big").render(max_chars=500)

    assert "truncated" in rendered
    assert "browser_find" in rendered


def test_long_labels_are_clipped() -> None:
    noisy = {"role": "WebArea", "children": [{"role": "button", "name": "x" * 2000}]}
    rendered = Builder().build(noisy, url="https://x.test", title="t").render()
    assert "…" in rendered
    assert len(rendered) < 1000


# -- the tools ---------------------------------------------------------------------


class Context:
    """A minimal ToolContext for driving the tools directly."""

    def __init__(self, *, policy: ApprovalPolicy | None = None) -> None:
        from harness_agentic.core.cancel import CancelToken
        from harness_agentic.envs.local import LocalEnvironment

        self.session_id = "s1"
        self.workspace_root = Path.cwd()
        self.cwd = Path.cwd()
        self.env = LocalEnvironment(Path.cwd())
        self.cancel = CancelToken()
        self.surface = "cli"
        self.messages: list[str] = []
        self._policy = policy or ApprovalPolicy(surface="cli", modes={"cli": Mode.ALLOW})

    def emit(self, message: str) -> None:
        self.messages.append(message)

    def approve(self, request: Any) -> bool:
        return self._policy.check(request).granted


def tools(fake: FakeDriver) -> ToolRegistry:
    return install_browser_tools(ToolRegistry(), fake)


def call(registry: ToolRegistry, name: str, ctx: Context, **arguments: Any) -> Any:
    tool = registry.get(name)
    return tool.handler(tool.validate_arguments(arguments), ctx)  # type: ignore[arg-type]


def test_navigating_returns_a_labelled_outline() -> None:
    fake = driver()
    registry = tools(fake)
    result = call(registry, "browser_navigate", Context(), url="https://example.test/login")

    assert not result.is_error
    assert "textbox" in result.text
    assert fake.visited == ["https://example.test/login"]
    # Page content is untrusted, exactly like a fetched page.
    assert "<untrusted-content" in result.text
    assert result.tainted


def test_an_internal_url_is_refused_before_the_browser_opens_it() -> None:
    # A browser is arbitrary code execution with a network connection, and it
    # will happily render the metadata service.
    fake = driver()
    result = call(tools(fake), "browser_navigate", Context(), url="http://169.254.169.254/latest/")
    assert result.is_error
    assert "metadata service" in result.text
    assert fake.visited == []


def test_clicking_needs_approval() -> None:
    # A click submits forms, confirms dialogs, and spends money.
    fake = driver()
    registry = tools(fake)
    call(registry, "browser_navigate", Context(), url="https://example.test/login")

    denied = call(
        registry,
        "browser_click",
        Context(policy=always_deny()),
        ref="e4",
        what="the Sign in button",
    )
    assert denied.is_error
    assert "not approved" in denied.text
    assert fake.clicked == []


def test_clicking_follows_a_link_and_returns_the_new_page() -> None:
    fake = driver()
    registry = tools(fake)
    call(registry, "browser_navigate", Context(), url="https://example.test/login")
    result = call(registry, "browser_click", Context(), ref="e4", what="Forgot password")

    assert not result.is_error
    assert "Welcome back" in result.text
    assert fake.current_url() == "https://example.test/home"


def test_a_stale_reference_clicks_nothing() -> None:
    fake = driver()
    registry = tools(fake)
    call(registry, "browser_navigate", Context(), url="https://example.test/login")
    # The page is re-read, so the old references are gone.
    call(registry, "browser_snapshot", Context())
    fake._refs.replace(99, {"e1": "e1"})  # simulate a generation the model has not seen

    result = call(registry, "browser_click", Context(), ref="e4", what="something")
    assert result.is_error
    assert "nothing was clicked" in result.text
    assert fake.clicked == []


def test_typing_shows_the_operator_what_is_being_submitted() -> None:
    seen: list[Any] = []

    def prompter(request: Any) -> bool:
        seen.append(request)
        return True

    fake = driver()
    registry = tools(fake)
    ctx = Context(
        policy=ApprovalPolicy(surface="cli", modes={"cli": Mode.PROMPT}, prompter=prompter)
    )
    call(registry, "browser_navigate", Context(), url="https://example.test/login")
    call(registry, "browser_type", ctx, ref="e1", text="someone@example.test", submit=True)

    assert fake.typed == [("e1", "someone@example.test", True)]
    # The text itself, not a preview: an operator approving a form submission
    # needs to see what is being submitted.
    assert seen[0].detail == "someone@example.test"


def test_find_answers_without_returning_the_whole_page() -> None:
    fake = driver()
    registry = tools(fake)
    call(registry, "browser_navigate", Context(), url="https://example.test/login")

    result = call(registry, "browser_find", Context(), query="password")
    assert not result.is_error
    assert "Password" in result.text
    assert "Remember me" not in result.text


def test_find_says_when_nothing_matches() -> None:
    fake = driver()
    registry = tools(fake)
    call(registry, "browser_navigate", Context(), url="https://example.test/login")
    result = call(registry, "browser_find", Context(), query="checkout")

    assert not result.is_error
    assert "Nothing interactive matches" in result.text


def test_a_screenshot_is_available_but_the_docstring_points_elsewhere() -> None:
    fake = driver()
    registry = tools(fake)
    call(registry, "browser_navigate", Context(), url="https://example.test/login")
    result = call(registry, "browser_screenshot", Context())

    assert result.images
    assert result.images[0].media_type == "image/png"
    assert fake.screenshots == 1
    # The tool exists, and it tells the model to prefer the outline.
    assert "Prefer browser_snapshot" in registry.get("browser_screenshot").description


def test_the_snapshot_tool_is_cheaper_than_the_screenshot_tool() -> None:
    # Not a micro-benchmark: the assertion is about which one the model is
    # steered towards, and reading is safe while clicking is not.
    registry = tools(driver())
    assert registry.get("browser_snapshot").danger is Danger.SAFE
    assert registry.get("browser_click").danger is Danger.WRITES
    assert registry.get("browser_navigate").danger is Danger.NETWORK


def test_acting_before_the_browser_starts_is_an_error_not_a_crash() -> None:
    fake = FakeDriver(policy=UrlPolicy(resolver=resolver))
    fake.add("https://example.test/x", LOGIN_PAGE)
    result = call(tools(fake), "browser_navigate", Context(), url="https://example.test/x")

    assert result.is_error
    assert "not running" in result.text


def test_a_missing_page_is_an_error() -> None:
    fake = driver()
    result = call(tools(fake), "browser_navigate", Context(), url="https://example.test/nope")
    assert result.is_error


def test_the_fake_driver_enforces_the_policy_too() -> None:
    # A fake that skips the policy hides exactly the bugs the policy is for.
    fake = FakeDriver(policy=UrlPolicy(resolver=resolver))
    fake.start()
    with pytest.raises(UrlRefused):
        fake.navigate("http://127.0.0.1:8080/admin")


def test_a_redirect_to_an_internal_address_is_refused() -> None:
    """The bug: only the URL the agent named was checked.

    A browser follows redirects itself, so an innocuous page that 302s to
    http://169.254.169.254/ was fetched and rendered with no second check -- and
    the metadata service hands IAM credentials to anything that asks. The policy
    has to be re-applied on every hop, which is the property this asserts of both
    drivers.
    """
    fake = FakeDriver(policy=UrlPolicy(resolver=resolver))
    fake.add("https://innocent.test/start", HOME_PAGE, redirect_to="http://169.254.169.254/")
    fake.start()

    with pytest.raises(UrlRefused, match="metadata"):
        fake.navigate("https://innocent.test/start")


def test_a_redirect_chain_is_bounded() -> None:
    # Otherwise a page redirecting to itself is an infinite loop, not a refusal.
    fake = FakeDriver(policy=UrlPolicy(resolver=resolver))
    fake.add("https://example.test/loop", HOME_PAGE, redirect_to="https://example.test/loop")
    fake.start()

    with pytest.raises(BrowserError, match="too many redirects"):
        fake.navigate("https://example.test/loop")


def test_a_redirect_to_a_permitted_page_is_followed() -> None:
    fake = FakeDriver(policy=UrlPolicy(resolver=resolver))
    fake.add("https://example.test/old", LOGIN_PAGE, redirect_to="https://example.test/home")
    fake.add("https://example.test/home", HOME_PAGE, title="Dashboard")
    fake.start()

    snapshot = fake.navigate("https://example.test/old")
    assert snapshot.title == "Dashboard"
    assert fake.current_url() == "https://example.test/home"


def test_every_request_is_judged_not_just_the_navigation() -> None:
    """Page script can reach an internal address without navigating at all.

    ``fetch('http://169.254.169.254/')`` from a loaded page is the same request
    as a navigation, so interception covers subresources too. This is the pure
    decision the real driver's route handler calls; the Playwright glue around it
    needs a browser.
    """
    from harness_agentic.browser.driver import request_allowed

    policy = UrlPolicy(resolver=resolver)
    assert request_allowed("https://example.test/app.js", policy)
    assert not request_allowed("http://169.254.169.254/latest/meta-data/", policy)
    assert not request_allowed("http://127.0.0.1:8080/admin", policy)
    # The page's own bytes reach nothing, so refusing them would break
    # navigation without denying any access.
    assert request_allowed("about:blank", policy)
    assert request_allowed("data:text/html,<p>hi</p>", policy)


def test_the_driver_protocol_is_satisfied_by_the_fake() -> None:
    from harness_agentic.browser.driver import Driver

    fake: Driver = FakeDriver()
    assert callable(fake.navigate)
    with pytest.raises(BrowserError):
        fake.snapshot()


def test_cdp_nodes_become_the_tree_the_builder_expects() -> None:
    """``page.accessibility`` was removed and the pin allows every version without it.

    So the toolset's cheapest and most-used tool raised ``AttributeError`` on any
    current install. The fallback reads the same data from CDP, which is what that
    API wrapped -- but CDP reports one flat list plus ``childIds``, and the tree
    builder wants the nesting.
    """
    from harness_agentic.browser.driver import _nest

    nodes = [
        {
            "nodeId": "1",
            "role": {"value": "RootWebArea"},
            "name": {"value": "Orders"},
            "childIds": ["2", "3"],
        },
        {
            "nodeId": "2",
            "role": {"value": "heading"},
            "name": {"value": "Order 1001"},
            "childIds": [],
        },
        {
            "nodeId": "3",
            "role": {"value": "generic"},
            "name": {"value": ""},
            "childIds": ["4"],
            "ignored": True,
        },
        {"nodeId": "4", "role": {"value": "button"}, "name": {"value": "Check"}, "childIds": []},
    ]

    tree = _nest(nodes)

    assert tree["role"] == "RootWebArea"
    assert tree["name"] == "Orders"
    assert [child["name"] for child in tree["children"]] == ["Order 1001", ""]
    # An ignored wrapper keeps its children; the builder collapses it later.
    wrapper = tree["children"][1]
    assert wrapper["role"] == ""
    assert wrapper["children"][0]["name"] == "Check"


def test_a_cycle_in_the_tree_does_not_recurse_forever() -> None:
    # A malformed tree can point back at itself, and a browser is untrusted input.
    from harness_agentic.browser.driver import _nest

    tree = _nest(
        [
            {"nodeId": "1", "role": {"value": "a"}, "name": {"value": ""}, "childIds": ["2"]},
            {"nodeId": "2", "role": {"value": "b"}, "name": {"value": ""}, "childIds": ["1"]},
        ]
    )
    assert tree["role"] in ("a", "b", "document")


def test_several_roots_are_wrapped_in_a_document() -> None:
    from harness_agentic.browser.driver import _nest

    tree = _nest(
        [
            {"nodeId": "1", "role": {"value": "a"}, "name": {"value": ""}, "childIds": []},
            {"nodeId": "9", "role": {"value": "b"}, "name": {"value": ""}, "childIds": []},
        ]
    )
    assert tree["role"] == "document"
    assert [child["role"] for child in tree["children"]] == ["a", "b"]
