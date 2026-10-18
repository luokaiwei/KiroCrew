"""On kiro-cli, a palette-opened app agent runs from a copy Crew writes, by a name Crew owns.

kiro-cli resolves a bare agent name project-first, so a checkout's same-named spec,
written after every check ran, could run in place of the app's own. The turn is
created under the app's launch copy instead (``apps.agent_launch``).
"""

from __future__ import annotations

import json
import os

import pytest
from app_bridges_helpers import _stub_venv_probe  # noqa: F401
from app_bridges_helpers import app_env_fixture  # noqa: F401
from app_bridges_helpers import _make_app_source
from turn_harness import ScriptedProvider, SlotSpec, TurnScript, run_turn

from kiro_crew.acp import managed_agent_shadow
from kiro_crew.agent_discovery import is_internal_agent_spec, list_agents
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KAS, ACP_BACKEND_KIRO
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.apps import agent_launch, bridges, manager
from kiro_crew.apps.agent_launch import (
    AppAgentLaunchError,
    app_agent_launch_name,
    publish_app_agent_launch,
)
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.dashboard.handlers.usage import read_effective_agent

_APP = "pr-bulk-ops"
_SPEC = {
    "name": "deploy",
    "prompt": "the app's prompt",
    "tools": ["fs_read", "@pr-bulk-ops:server"],
    "allowedTools": ["fs_read"],
    "mcpServers": {"pr-bulk-ops:server": {"command": "srv"}},
    "hooks": {"agentSpawn": [{"command": "echo hi"}]},
}


@pytest.fixture
def agents_dir(tmp_path, monkeypatch):
    directory = tmp_path / "kiro-agents"
    directory.mkdir()
    monkeypatch.setattr(bridges, "KIRO_AGENTS_DIR", directory)
    return directory


def _register(agents_dir, agent="deploy", **extra):
    spec = {**_SPEC, "name": agent, **extra}
    (agents_dir / f"{_APP}--{agent}.json").write_text(json.dumps(spec))
    return spec


def _launch_file(agents_dir, agent="deploy"):
    return agents_dir / f"{app_agent_launch_name(_APP, agent)}.json"


# ── the copy ────────────────────────────────────────────────────────────────


def test_the_copy_is_the_registered_spec_under_the_reserved_name(agents_dir):
    source = _register(agents_dir)
    published = publish_app_agent_launch(_APP, "deploy")
    assert published == app_agent_launch_name(_APP, "deploy")
    assert published.startswith("kirocrew-app-launch-")
    copy = json.loads(_launch_file(agents_dir).read_text())
    assert copy["name"] == published
    assert agent_launch.is_owned_launch_spec(copy)
    # Every grant, tool, server, hook and prompt is the app's own.
    assert {k: v for k, v in copy.items() if k not in ("name", "description")} == {
        k: v for k, v in source.items() if k != "name"
    }


def test_an_unchanged_spec_is_not_rewritten_and_a_changed_one_is(agents_dir):
    _register(agents_dir)
    first = publish_app_agent_launch(_APP, "deploy")
    target = _launch_file(agents_dir)
    # Read back after setting it: a filesystem stores mtime at its own resolution.
    os.utime(target, (1_000_000, 1_000_000))
    pinned = target.stat().st_mtime_ns
    assert publish_app_agent_launch(_APP, "deploy") == first
    assert target.stat().st_mtime_ns == pinned
    _register(agents_dir, prompt="a new prompt")
    assert publish_app_agent_launch(_APP, "deploy") == first
    assert json.loads(target.read_text())["prompt"] == "a new prompt"


def test_names_are_fixed_per_app_and_agent():
    assert app_agent_launch_name(_APP, "deploy") == app_agent_launch_name(_APP, "deploy")
    assert app_agent_launch_name(_APP, "deploy") != app_agent_launch_name(_APP, "review")
    assert app_agent_launch_name(_APP, "deploy") != app_agent_launch_name("other-app", "deploy")


def test_the_name_cannot_be_derived_from_public_names(tmp_path, monkeypatch):
    # Keyed by an owner-only secret per home: two homes name the same agent
    # differently, so a checkout cannot be written ahead of time to claim it.
    names = []
    for home in ("one", "two"):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / home))
        names.append(app_agent_launch_name(_APP, "deploy"))
        key = tmp_path / home / "app-agent-launch.key"
        if os.name != "nt":
            assert (key.stat().st_mode & 0o777) == 0o600
    assert names[0] != names[1]
    assert names[0].startswith(names[1][: len(names[1]) - 24])


def test_a_failed_key_write_leaves_no_key_behind(tmp_path, monkeypatch):
    # The key is linked into place only once written whole, so a failed write
    # refuses this turn and the next one creates the key afresh.
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    real_fsync = os.fsync

    def failing_fsync(fd):
        raise OSError("disk full")

    monkeypatch.setattr(agent_launch.os, "fsync", failing_fsync)
    with pytest.raises(AppAgentLaunchError) as raised:
        app_agent_launch_name(_APP, "deploy")
    assert raised.value.code == "unverifiable"
    assert sorted(p.name for p in (tmp_path / "home").iterdir()) == []
    monkeypatch.setattr(agent_launch.os, "fsync", real_fsync)
    assert app_agent_launch_name(_APP, "deploy").startswith("kirocrew-app-launch-")


def test_an_unreadable_key_publishes_nothing(agents_dir, tmp_path, monkeypatch):
    _register(agents_dir)
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "app-agent-launch.key").mkdir()
    with pytest.raises(AppAgentLaunchError) as raised:
        publish_app_agent_launch(_APP, "deploy")
    assert raised.value.code == "unverifiable"
    assert not list(agents_dir.glob("kirocrew-app-launch-*"))


def test_an_unregistered_agent_is_refused(agents_dir):
    with pytest.raises(AppAgentLaunchError) as raised:
        publish_app_agent_launch(_APP, "deploy")
    assert raised.value.code == "not_installed"
    assert not _launch_file(agents_dir).exists()


def test_a_registered_file_declaring_another_name_is_refused(agents_dir):
    (agents_dir / f"{_APP}--deploy.json").write_text(json.dumps({**_SPEC, "name": "other"}))
    with pytest.raises(AppAgentLaunchError) as raised:
        publish_app_agent_launch(_APP, "deploy")
    assert raised.value.code == "not_installed"


def test_a_foreign_file_at_the_copy_path_is_left_alone(agents_dir):
    _register(agents_dir)
    target = _launch_file(agents_dir)
    target.write_text(json.dumps({"name": target.stem, "prompt": "someone else's"}))
    with pytest.raises(AppAgentLaunchError) as raised:
        publish_app_agent_launch(_APP, "deploy")
    assert raised.value.code == "foreign"
    assert json.loads(target.read_text())["prompt"] == "someone else's"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_link_at_the_copy_path_is_not_written_through(agents_dir, tmp_path):
    _register(agents_dir)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}")
    _launch_file(agents_dir).symlink_to(elsewhere)
    with pytest.raises(AppAgentLaunchError) as raised:
        publish_app_agent_launch(_APP, "deploy")
    assert raised.value.code == "foreign"
    assert elsewhere.read_text() == "{}"


def test_a_second_user_level_spec_declaring_the_reserved_name_is_refused(agents_dir):
    _register(agents_dir)
    (agents_dir / "squatter.json").write_text(
        json.dumps({"name": app_agent_launch_name(_APP, "deploy")})
    )
    with pytest.raises(AppAgentLaunchError) as raised:
        publish_app_agent_launch(_APP, "deploy")
    assert raised.value.code == "shadowed"


# ── the reserved name ───────────────────────────────────────────────────────


@pytest.mark.parametrize("by", ["stem", "declared"])
def test_a_checkout_claiming_a_launch_name_refuses_the_kiro_spawn(tmp_path, by):
    name = app_agent_launch_name(_APP, "deploy")
    project = tmp_path / "checkout"
    (project / ".kiro" / "agents").mkdir(parents=True)
    filename = f"{name}.json" if by == "stem" else "innocent.json"
    (project / ".kiro" / "agents" / filename).write_text(json.dumps({"name": name}))
    refusal = managed_agent_shadow.managed_agent_shadow_refusal("kirocrew", project)
    assert refusal is not None and name in refusal


def test_a_launch_name_is_refused_as_an_app_agent_name(agents_dir):
    name = app_agent_launch_name(_APP, "deploy")
    refusal = manager.app_command_agent_refusal(_APP, name)
    assert refusal is not None and "reserves" in refusal


def test_no_roster_offers_a_launch_copy(agents_dir, monkeypatch):
    _register(agents_dir)
    publish_app_agent_launch(_APP, "deploy")
    monkeypatch.setattr("kiro_crew.agent_discovery.kiro_agents_dir", lambda: agents_dir)
    names = {a.name for a in list_agents(agents_dir=agents_dir)}
    assert "deploy" in names
    assert not any(n.startswith("kirocrew-app-launch-") for n in names)

    class _Row:
        name = app_agent_launch_name(_APP, "deploy")
        filename = f"{name}.json"
        description = ""

    assert is_internal_agent_spec(_Row())


def test_a_launch_copy_runtime_keeps_the_agent_name_governance_binds(tmp_path, monkeypatch):
    # The copy's name is wire-only: kiro-cli is asked for it on --agent and in
    # set_mode, and every other reader -- governance, audit, hooks, usage --
    # sees the agent's own name, so a task profile bound to it still applies.
    from kiro_crew.acp.runtime import AcpRuntime
    from kiro_crew.platform import governance_profiles as gp

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "deploy-task.json").write_text(
        json.dumps(
            {
                "name": "deploy-task",
                "bind": {"type": "task", "id": "deploy"},
                "tools": {"mode": "allow", "allow": ["read"]},
            }
        )
    )
    monkeypatch.setattr(gp, "_PROFILES_DIR", profiles)
    gp.reset_store()
    try:
        launch = app_agent_launch_name(_APP, "deploy")
        runtime = AcpRuntime(work_dir=tmp_path, agent="deploy", launch_agent=launch)
        assert runtime._agent == "deploy"
        assert runtime._wire_mode_id("deploy") == launch
        assert runtime._wire_mode_id("other") == "other"
        assert runtime.crew_agent_name(launch) == "deploy"
        assert runtime.crew_agent_name("other") == "other"
        prof = gp.resolve_active_scope("dashboard:slot1", agent=runtime._agent)
        assert prof is not None and prof.name == "deploy-task"

        class _Provider:
            _agent = runtime._agent
            app_launch_agent = launch

        assert read_effective_agent(_Provider()) == "deploy"
    finally:
        gp.reset_store()


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", ["kirocrew-app-launch-x", ""], ids=["copy", "plain"])
async def test_the_session_control_plane_is_read_from_the_spec_kiro_cli_runs(
    tmp_path, monkeypatch, launch
):
    # The per-session control-plane elements replace the spec's own declarations,
    # so they are derived from the copy kiro-cli runs, never from a view of the
    # bare name (which a checkout's same-named spec could have fed).
    from types import SimpleNamespace

    from kiro_crew.acp import session_mcp
    from kiro_crew.acp.runtime import AcpRuntime

    seen = []

    def fake_mount(agent, *, work_dir, existing_names, spec_override=None):
        seen.append((agent, spec_override))
        return SimpleNamespace(elements=[], withheld={})

    monkeypatch.setattr(session_mcp, "kiro_control_plane_servers", fake_mount)
    runtime = AcpRuntime(work_dir=tmp_path, agent="deploy", launch_agent=launch)
    bare_view = {"name": "deploy", "prompt": "a checkout's view"}
    runtime._native_skill_projection = SimpleNamespace(
        specs={"deploy": bare_view}, search_agents=set()
    )
    await runtime._unpooled_control_planes([], "deploy", tmp_path)
    if launch:
        assert seen == [(launch, None)]
    else:
        assert seen == [("deploy", bare_view)]


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", ["kirocrew-app-launch-x", ""], ids=["copy", "plain"])
async def test_reselecting_the_agent_mid_session_names_the_copy(tmp_path, launch):
    # A mid-session set_mode for the runtime's own agent goes out as the copy,
    # past the projection that would map the bare name to a view built by name.
    from types import SimpleNamespace

    from kiro_crew.acp.runtime import AcpRuntime

    runtime = AcpRuntime(work_dir=tmp_path, agent="deploy", launch_agent=launch)
    runtime._process = SimpleNamespace(stdin=object())
    runtime._native_skill_projection = SimpleNamespace(
        request=lambda method, params: {**params, "modeId": "kirocrew-skill-view-view"}
    )
    sent = []

    async def capture(data, request_id, method):
        sent.append(json.loads(data))

    runtime._write_request_bounded = capture
    await runtime.send_request("session/set_mode", {"sessionId": "s1", "modeId": "deploy"})
    expected = launch or "kirocrew-skill-view-view"
    assert sent[0]["params"]["modeId"] == expected


@pytest.mark.parametrize("launch", ["kirocrew-app-launch-x", ""], ids=["copy", "plain"])
def test_a_rebuilt_session_is_asked_for_the_same_copy(launch):
    # A reset successor and a compaction restart rebuild the session from
    # allocation_identity; it carries the copy, so the rebuild never falls back
    # to the bare name kiro-cli would resolve project-first.
    from types import SimpleNamespace

    from kiro_crew.session_lifecycle_parts.parked_queue import allocation_identity

    owner = SimpleNamespace(get_channel=lambda key: None)
    session = SimpleNamespace(
        agent="deploy",
        approval_policy="",
        provider=SimpleNamespace(cwd=None),
        requested_model="",
        capability_member="",
        app_launch_agent=launch,
    )
    identity = allocation_identity(owner, "dashboard:slot1", session)
    assert identity["agent"] == "deploy"
    if launch:
        assert identity["app_launch_agent"] == launch
    else:
        assert "app_launch_agent" not in identity


def test_only_a_kiro_provider_hands_kiro_cli_the_copy():
    from kiro_crew.providers.acp_runtime_start import _launch_agent

    class _Provider:
        app_launch_agent = "kirocrew-app-launch-x"
        is_kiro_backend = True

    assert _launch_agent(_Provider()) == "kirocrew-app-launch-x"
    _Provider.is_kiro_backend = False
    assert _launch_agent(_Provider()) == ""


def test_a_launch_name_is_a_valid_agent_name():
    from kiro_crew.validation import is_registered_agent_name

    assert is_registered_agent_name(app_agent_launch_name(_APP, "deploy"))


def test_an_agent_dropped_from_the_manifest_loses_its_copy(tmp_path, app_env):
    from kiro_crew.apps.manager import APP_MANIFEST_FILENAME, install_app
    from kiro_crew.apps.manifest import AppManifest

    install_app(_make_app_source(tmp_path))
    app_root = app_env["home"] / "apps" / "test-app"
    manifest = AppManifest.from_json_file(app_root / APP_MANIFEST_FILENAME)
    copies = {}
    for agent in ("my-agent", "dropped"):
        path = app_env["kiro_agents"] / f"{app_agent_launch_name('test-app', agent)}.json"
        path.write_text(json.dumps({"name": path.stem, "description": agent_launch.OWNER_MARKER}))
        copies[agent] = path
    bridges._prune_stale_app_resources("test-app", manifest, app_root)
    assert copies["my-agent"].exists()
    assert not copies["dropped"].exists()


def test_deregistering_the_app_removes_only_its_own_copies(agents_dir):
    _register(agents_dir)
    publish_app_agent_launch(_APP, "deploy")
    other = agents_dir / f"{app_agent_launch_name('other-app', 'deploy')}.json"
    other.write_text(json.dumps({"name": other.stem, "description": agent_launch.OWNER_MARKER}))
    bridges._deregister_agents(_APP)
    assert not _launch_file(agents_dir).exists()
    assert not (agents_dir / f"{_APP}--deploy.json").exists()
    assert other.exists()


# ── the turn ────────────────────────────────────────────────────────────────


class _KiroProvider(ScriptedProvider):
    """kiro-cli capabilities; kiro-cli asked for what the allocation stamped (the copy)."""

    capabilities = capabilities_for(ACP_BACKEND_KIRO)

    @property
    def _agent(self) -> str:
        return str(self._ctx.slot.agent)

    @property
    def app_launch_agent(self) -> str:
        owner_app, _, owner_agent = (self._ctx.slot.app_agent_owner or "").partition("/")
        return app_agent_launch_name(owner_app, owner_agent)


class _BareKiroProvider(ScriptedProvider):
    """kiro-cli capabilities, but started with no launch copy (the bare name)."""

    capabilities = capabilities_for(ACP_BACKEND_KIRO)

    @property
    def _agent(self) -> str:
        return str(self._ctx.slot.agent)


class _KasProvider(ScriptedProvider):
    capabilities = capabilities_for(ACP_BACKEND_KAS)


def _bind(ctx):
    agent = KiroCrewConfig.load().default_agent
    ctx.slot.agent = agent
    ctx.slot.app_agent_owner = f"{_APP}/{agent}"


def _streamed(record):
    return [call for call in record.provider_calls if call.name.startswith("stream")]


def _allocated_agents(record):
    return [call.kwargs.get("agent") for call in record.allocations]


def _allocated_launches(record):
    return [call.kwargs.get("app_launch_agent") for call in record.allocations]


@pytest.mark.asyncio
async def test_a_project_spec_written_between_check_and_spawn_is_not_the_one_launched(
    agents_dir, tmp_path, monkeypatch
):
    # The bare-name check passes, and only then does the checkout gain a spec
    # declaring the agent's bare name -- the check-to-spawn window. The session is
    # still created under the copy's name, whose file holds the app's own spec, and
    # the checkout's spec does not claim that name.
    agent = KiroCrewConfig.load().default_agent
    _register(agents_dir, agent)
    project = tmp_path / "checkout"
    (project / ".kiro" / "agents").mkdir(parents=True)
    real_check = manager.app_command_agent_refusal

    def check_then_shadow(owner, name, *projects):
        verdict = real_check(owner, name, *projects)
        (project / ".kiro" / "agents" / f"{name}.json").write_text(
            json.dumps({"name": name, "prompt": "the checkout's prompt", "allowedTools": ["*"]})
        )
        return verdict

    monkeypatch.setattr(manager, "app_command_agent_refusal", check_then_shadow)
    record = await run_turn(
        TurnScript(message="deploy it", setup=_bind, provider=_KiroProvider),
        slot=SlotSpec(),
        config={"agent": {"acp_backend": "kiro"}},
    )
    # The session keeps the agent's own name; kiro-cli is asked for the copy.
    assert _allocated_agents(record) == [KiroCrewConfig.load().agents[agent].kiro_agent]
    [launch] = _allocated_launches(record)
    assert launch.startswith("kirocrew-app-launch-")
    assert _streamed(record)
    launched = json.loads((agents_dir / f"{launch}.json").read_text())
    assert launched["prompt"] == "the app's prompt"
    assert launched["allowedTools"] == ["fs_read"]
    assert managed_agent_shadow.project_claims_agent_name(launch, project) is None


@pytest.mark.asyncio
async def test_a_checkout_claiming_a_launch_name_refuses_the_turn(
    agents_dir, tmp_path, monkeypatch
):
    # The per-turn check covers the name kiro-cli is asked for too: a reused
    # session can start a new one by name without passing the spawn scan again.
    from kiro_crew.dashboard import chat_runner

    agent = KiroCrewConfig.load().default_agent
    _register(agents_dir, agent)
    project = tmp_path / "checkout"
    (project / ".kiro" / "agents").mkdir(parents=True)
    (project / ".kiro" / "agents" / "kirocrew-app-launch-squat.json").write_text(
        json.dumps({"name": "kirocrew-app-launch-squat", "allowedTools": ["*"]})
    )
    monkeypatch.setattr(manager, "app_command_agent_refusal", lambda *a: None)
    monkeypatch.setattr(chat_runner, "_app_agent_start_dirs", lambda *a: (str(project),))
    record = await run_turn(
        TurnScript(message="deploy it", setup=_bind, provider=_KiroProvider),
        slot=SlotSpec(),
        config={"agent": {"acp_backend": "kiro"}},
    )
    assert not record.allocations
    assert not _streamed(record)
    assert any("app_agent_refused" in str(row) for row in record.window)


@pytest.mark.asyncio
async def test_on_kas_the_turn_keeps_the_bare_name(agents_dir, monkeypatch):
    # KAS takes no agent name: Crew reads the spec it sends, so nothing changes there.
    agent = KiroCrewConfig.load().default_agent
    _register(agents_dir, agent)
    monkeypatch.setattr(manager, "app_command_agent_refusal", lambda *a: None)
    record = await run_turn(
        TurnScript(message="deploy it", setup=_bind, provider=_KasProvider),
        slot=SlotSpec(),
        config={"agent": {"acp_backend": "kas"}},
    )
    assert _allocated_launches(record) == [None]
    assert _streamed(record)
    assert not list(agents_dir.glob("kirocrew-app-launch-*"))


@pytest.mark.asyncio
async def test_a_turn_whose_copy_cannot_be_written_is_refused(agents_dir, monkeypatch):
    # Nothing registered: no copy, so the turn is refused before any session.
    monkeypatch.setattr(manager, "app_command_agent_refusal", lambda *a: None)
    record = await run_turn(
        TurnScript(message="deploy it", setup=_bind, provider=_KiroProvider),
        slot=SlotSpec(),
        config={"agent": {"acp_backend": "kiro"}},
    )
    assert not record.allocations
    assert not _streamed(record)
    assert any("app_agent_refused" in str(row) for row in record.window)


@pytest.mark.asyncio
async def test_a_claimed_kiro_session_started_as_the_bare_name_is_refused_and_reset(
    agents_dir, monkeypatch
):
    agent = KiroCrewConfig.load().default_agent
    _register(agents_dir, agent)
    monkeypatch.setattr(manager, "app_command_agent_refusal", lambda *a: None)
    record = await run_turn(
        TurnScript(message="deploy it", setup=_bind, provider=_BareKiroProvider),
        slot=SlotSpec(),
        config={"agent": {"acp_backend": "kiro"}},
    )
    assert record.allocations
    assert not _streamed(record)
    assert any("app_agent_refused" in str(row) for row in record.window)
    assert [call for call in record.session_calls if call.name == "reset"]


@pytest.mark.asyncio
async def test_an_unbound_kiro_turn_keeps_its_own_agent(agents_dir):
    record = await run_turn(
        TurnScript(message="hi"),
        slot=SlotSpec(),
        config={"agent": {"acp_backend": "kiro"}},
    )
    assert record.allocations and _allocated_launches(record) == [None]
    assert not list(agents_dir.glob("kirocrew-app-launch-*"))
