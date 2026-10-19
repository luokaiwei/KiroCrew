"""The dashboard -> Slack mirror: its prompt echo and its "Thinking…" placeholder.

Two defects a linked thread showed on every mirrored turn:

* the prompt echo escaped every ``<…>`` entity and wrapped the prompt in ``_…_``,
  so a pasted ``<https://…|label>`` link or ``<#C…>`` channel reference arrived as
  raw text. The echo now keeps those live, flattens a ``<@U…>`` mention to plain
  ``@…`` so it never re-pings anyone, still escapes ``<!channel>`` and friends and
  any stray or truncated bracket, and is posted as a ``> 💬 …`` quote;
* the mirror's task stream opened with a "Thinking…" line, and a stream only
  appends, so that line sat above Slack's own "Thinking completed" plan header for
  good. The line is now its own message, deleted when the stream, the answer or
  the turn's end takes over.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from turn_harness import Do, Raise, TurnContext, TurnScript, run_turn

from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_CALL,
    STOP_REASON_END_TURN,
    AcpEvent,
)
from kiro_crew.dashboard.chat_runner import _prepare_mirror_msg
from kiro_crew.slack.client import SlackClientOps
from kiro_crew.slack.format import escape_mrkdwn_keep_links, mirror_prompt_echo

_DONE = AcpEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)
_ANSWER = AcpEvent(kind=EVENT_TEXT_CHUNK, text="the reply")
_TOOL = AcpEvent(kind=EVENT_TOOL_CALL, tool_call_id="t1", title="read_1", tool_kind="read")


class _SlackRecorder:
    """A Slack client recording each call, bound to the real port's signatures.

    Each post returns a distinct ts, so a delete can be matched to what it removed.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        declared = getattr(SlackClientOps, name, None) if not name.startswith("_") else None
        if not inspect.iscoroutinefunction(declared):
            raise AttributeError(name)
        signature = inspect.signature(declared)

        async def _call(*args: Any, **kwargs: Any) -> str:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.calls.append((name, dict(bound.arguments)))
            return f"{name}-ts-{len(self.calls)}"

        return _call

    def posts(self) -> list[tuple[str, str]]:
        """(ts, text) of every post_message, in order."""
        return [
            (f"post_message-ts-{i + 1}", args["text"])
            for i, (name, args) in enumerate(self.calls)
            if name == "post_message"
        ]


class _StreamRefusingSlack(_SlackRecorder):
    """Slack refuses chat.startStream (an org-wide install with no recipient)."""

    def __getattr__(self, name: str) -> Any:
        call = super().__getattr__(name)
        if name != "start_stream":
            return call

        async def _refused(*args: Any, **kwargs: Any) -> None:
            await call(*args, **kwargs)
            return None

        return _refused


async def _linked_turn(
    events: list[Any], message: str = "hello", slack: _SlackRecorder | None = None
) -> _SlackRecorder:
    slack = slack or _SlackRecorder()

    def _linked(ctx: TurnContext) -> None:
        ctx.state.slack_client = slack
        ctx.state.sessions.set_slack_link("dashboard:chat-1", "thread-1", "C-1")

    await run_turn(TurnScript(events=events, message=message, setup=_linked))
    return slack


# ── The echo: what survives, what is neutralised ─────────────────────────────


class TestEchoKeepsLinks:
    def test_a_web_link_and_its_label_stay_live(self) -> None:
        assert escape_mrkdwn_keep_links("see <https://example.com/a_b|the doc>") == (
            "see <https://example.com/a_b|the doc>"
        )
        assert escape_mrkdwn_keep_links("<https://example.com>") == "<https://example.com>"

    def test_a_channel_reference_stays_live(self) -> None:
        assert escape_mrkdwn_keep_links("in <#C0123ABC>") == "in <#C0123ABC>"
        assert escape_mrkdwn_keep_links("<#C0123ABC|general>") == "<#C0123ABC|general>"

    def test_a_label_ampersand_is_escaped(self) -> None:
        assert escape_mrkdwn_keep_links("<https://x.com|a & b>") == "<https://x.com|a &amp; b>"


class TestEchoNeutralises:
    @pytest.mark.parametrize(
        ("mention", "flat"),
        [
            ("<@U0123ABC>", "@U0123ABC"),
            ("<@W0123ABC>", "@W0123ABC"),
            ("<@U0123ABC|kyle>", "@kyle"),
        ],
    )
    def test_a_user_mention_becomes_plain_text(self, mention: str, flat: str) -> None:
        out = escape_mrkdwn_keep_links(f"ping {mention} now")
        assert out == f"ping {flat} now"
        assert "<@" not in out

    @pytest.mark.parametrize(
        "broadcast",
        [
            "<!channel>",
            "<!here>",
            "<!everyone>",
            "<!subteam^S0123|@team>",
            "<mailto:a@b.co>",
            # A broadcast nested in a kept shape is matched on its own, not kept.
            "<https://x.com|<!channel>>",
            "<#C0123ABC|<!here>>",
            "<@U0123ABC|<!everyone>>",
        ],
    )
    def test_a_broadcast_or_other_entity_is_escaped(self, broadcast: str) -> None:
        out = escape_mrkdwn_keep_links(f"hey {broadcast}")
        assert "<" not in out and ">" not in out
        assert out.startswith("hey &lt;") and out.endswith("&gt;")

    def test_stray_brackets_and_ampersands_are_escaped(self) -> None:
        assert escape_mrkdwn_keep_links("a < b && c > d") == "a &lt; b &amp;&amp; c &gt; d"

    def test_a_link_the_truncation_cut_off_is_escaped(self) -> None:
        # The echo bound is 500 chars; a link straddling it loses its ``>``.
        cut = _prepare_mirror_msg("x" * 470 + " <https://example.com/very/long/path|label>")
        assert len(cut) == 500 and not cut.endswith(">")
        out = escape_mrkdwn_keep_links(cut)
        assert "<" not in out
        assert out.endswith(" &lt;https://example.com/very/lon")

    def test_an_unterminated_opener_cannot_capture_a_later_entity(self) -> None:
        out = escape_mrkdwn_keep_links("<https://a.com <!channel>")
        assert "<!channel>" not in out
        assert out == "&lt;https://a.com &lt;!channel&gt;"

    def test_an_exfiltration_url_stays_redacted_and_is_not_a_link(self) -> None:
        payload = "QUtJQUlPU0ZPRE5ON0VYQU1QTEVrZXlzZWNyZXRkYXRhMTIzNDU2Nzg5MA"
        raw = f"open <https://evil.example.com/c?d={payload}|harmless> please"
        out = escape_mrkdwn_keep_links(_prepare_mirror_msg(raw))
        assert payload not in out
        assert "evil.example.com/c" not in out
        assert "REDACTED" in out
        assert "<" not in out


class TestEchoIsAQuote:
    def test_every_line_is_quoted_and_a_leading_gt_cannot_nest(self) -> None:
        assert mirror_prompt_echo("first\n> second") == "> 💬 first\n> &gt; second"

    def test_no_italic_wrapper(self) -> None:
        assert mirror_prompt_echo("<https://a.com/x_y|z>") == "> 💬 <https://a.com/x_y|z>"


@pytest.mark.asyncio
async def test_a_mirrored_turn_echoes_links_live_and_pings_nobody() -> None:
    slack = await _linked_turn(
        [_ANSWER, _DONE],
        message="look at <https://example.com/a_b|the doc> in <#C0123ABC> cc <@U0123ABC> <!here>",
    )
    [echo] = [text for _, text in slack.posts() if "💬" in text]
    assert echo == (
        "> 💬 look at <https://example.com/a_b|the doc> in <#C0123ABC> cc @U0123ABC &lt;!here&gt;"
    )


# ── The "Thinking…" line never outlives its turn ─────────────────────────────


def _thinking_posts(slack: _SlackRecorder) -> list[str]:
    return [ts for ts, text in slack.posts() if "Thinking" in text]


def _deleted(slack: _SlackRecorder) -> list[str]:
    return [args["ts"] for name, args in slack.calls if name == "delete_message"]


def _stream_texts(slack: _SlackRecorder) -> list[str]:
    return [args.get("initial_text") or "" for name, args in slack.calls if name == "start_stream"]


def _surviving_thinking(slack: _SlackRecorder) -> list[str]:
    """Every "Thinking" line still in the thread once the turn has ended.

    A posted placeholder survives unless it was deleted; a line written into the
    append-only task stream always survives.
    """
    deleted = set(_deleted(slack))
    kept = [text for ts, text in slack.posts() if "Thinking" in text and ts not in deleted]
    return kept + [text for text in _stream_texts(slack) if "Thinking" in text]


@pytest.mark.asyncio
async def test_a_tool_turn_deletes_the_thinking_line_when_the_stream_opens() -> None:
    slack = await _linked_turn([_TOOL, _ANSWER, _DONE])
    assert _surviving_thinking(slack) == []
    names = [name for name, _ in slack.calls]
    [thinking] = _thinking_posts(slack)
    assert _deleted(slack) == [thinking]
    # The placeholder goes as the stream takes over: after it opens, before its
    # first task, and so before the answer is posted.
    delete_at = names.index("delete_message")
    assert names.index("start_stream") < delete_at < names.index("append_task")
    answer_at = max(i for i, (n, a) in enumerate(slack.calls) if n == "post_message")
    assert delete_at < answer_at
    assert names.count("stop_stream") == 1


@pytest.mark.asyncio
async def test_a_tool_free_turn_deletes_the_thinking_line_before_the_answer() -> None:
    slack = await _linked_turn([_ANSWER, _DONE])
    assert _surviving_thinking(slack) == []
    names = [name for name, _ in slack.calls]
    [thinking] = _thinking_posts(slack)
    assert _deleted(slack) == [thinking]
    assert "start_stream" not in names
    [answer_at] = [
        i
        for i, (n, a) in enumerate(slack.calls)
        if n == "post_message" and a["text"] == "the reply"
    ]
    assert names.index("delete_message") < answer_at


@pytest.mark.asyncio
async def test_a_failed_turn_still_deletes_the_thinking_line() -> None:
    slack = await _linked_turn([Raise(RuntimeError("boom"))])
    assert _surviving_thinking(slack) == []
    [thinking] = _thinking_posts(slack)
    assert _deleted(slack) == [thinking]


@pytest.mark.asyncio
async def test_the_stream_opens_once_and_the_line_goes_once_over_many_tools() -> None:
    second = AcpEvent(kind=EVENT_TOOL_CALL, tool_call_id="t2", title="read_2", tool_kind="read")
    slack = await _linked_turn([_TOOL, second, _ANSWER, _DONE])
    names = [name for name, _ in slack.calls]
    assert names.count("start_stream") == 1
    assert len(_deleted(slack)) == 1
    assert _surviving_thinking(slack) == []


@pytest.mark.asyncio
async def test_a_refused_stream_leaves_the_line_up_until_the_answer() -> None:
    slack = _StreamRefusingSlack()
    marks: list[int] = []
    second = AcpEvent(kind=EVENT_TOOL_CALL, tool_call_id="t2", title="read_2", tool_kind="read")
    tools_ran = Do(lambda ctx: marks.append(len(slack.calls)))
    await _linked_turn([_TOOL, second, tools_ran, _ANSWER, _DONE], slack=slack)
    names = [name for name, _ in slack.calls]
    assert names.count("start_stream") == 1
    assert "append_task" not in names
    # Nothing deleted the line while the tools ran.
    [mark] = marks
    assert "delete_message" not in names[:mark]
    # Still working while tools run, then gone before the answer lands.
    [answer_at] = [
        i
        for i, (n, a) in enumerate(slack.calls)
        if n == "post_message" and a["text"] == "the reply"
    ]
    assert names.index("start_stream") < names.index("delete_message") < answer_at
    assert _surviving_thinking(slack) == []


@pytest.mark.asyncio
async def test_a_withheld_turn_opens_no_stream_and_still_retires_the_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kiro_crew.dashboard import chat_runner

    monkeypatch.setattr(chat_runner, "cross_surface_withheld", lambda state, slot: True)
    slack = await _linked_turn([_TOOL, _ANSWER, _DONE])
    names = [name for name, _ in slack.calls]
    assert "start_stream" not in names
    assert "append_task" not in names
    assert _surviving_thinking(slack) == []
