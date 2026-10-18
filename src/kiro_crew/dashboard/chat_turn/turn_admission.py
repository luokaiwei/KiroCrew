"""Whether a dashboard turn may run on the session it is bound to: the current binding and its
requirement, the owner check and the dispatch lock's release."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiro_crew.dashboard.chat_runner import (
        ACP_BACKEND_KIRO,
        _MemoryUnavailable,
        effective_session_key,
    )
    from kiro_crew.dashboard.chat_turn.app_agent_binding import _app_agent_start_dirs


def _current_binding(turn) -> tuple:
    slot = turn.slot
    return (
        slot.agent,
        slot.project,
        slot.workspace,
        slot.memory_store,
        slot._memory_assignment_from_history,
        effective_session_key(slot),
        slot.agent_kind,
    )


def _require_current_binding(turn, *, project_may_be_staged: bool = False) -> None:
    # ``project_may_be_staged``: once the session is registered the
    # project route may stage a pick for the NEXT turn (it writes
    # slot.project and arms the deferred reset). Checks made after that
    # point compare every other binding field and leave the project
    # out, because this turn runs on its start-of-turn project and the
    # staged pick is not a change to it.
    registered_slot = turn.registered_slot
    slot = turn.slot
    state = turn.state
    current = _current_binding(turn)
    expected = turn.selected_binding
    if project_may_be_staged:
        current = current[:1] + current[2:]
        expected = expected[:1] + expected[2:]
    if (registered_slot is slot and state._slots.get(slot.key) is not slot) or current != expected:
        raise _MemoryUnavailable(
            "memory_unavailable: conversation binding changed during preparation; " "retry the turn"
        )


def _check_owner(turn) -> tuple[str | None, str]:
    """The refusal of an app-owned turn (None when it may run) and the launch copy it runs as.

    The copy's name is empty unless the turn runs on kiro-cli.
    """
    # Only kiro-cli and KAS launch the app's named spec; checked first,
    # since the name check below means nothing on another backend.
    from kiro_crew.apps.agent_launch import AppAgentLaunchError, publish_app_agent_launch
    from kiro_crew.apps.manager import (
        app_agent_backend_refusal,
        app_agent_launch_refusal,
        app_command_agent_refusal,
    )

    _owner_backend = turn._owner_backend
    _owner_host_allowed = turn._owner_host_allowed
    _owner_project = turn._owner_project
    owner_agent = turn.owner_agent
    owner_app = turn.owner_app
    session_key = turn.session_key
    state = turn.state
    if not _owner_host_allowed:
        return app_agent_backend_refusal(owner_app), ""
    # Both halves touch the filesystem (the work-dir root, the specs).
    dirs = _app_agent_start_dirs(state, session_key, _owner_project)
    refused = app_command_agent_refusal(owner_app, owner_agent, *dirs)
    if refused:
        return refused, ""
    if _owner_backend == ACP_BACKEND_KIRO:
        try:
            launch = publish_app_agent_launch(owner_app, owner_agent)
        except AppAgentLaunchError as exc:
            return app_agent_launch_refusal(owner_app, exc.detail), ""
        # The name kiro-cli resolves for this turn is the copy's, so the
        # same per-turn project check covers it too: a reused session can
        # start a new one by name without passing the spawn scan again.
        from kiro_crew.agent_sdk import reserved_agent_name_refusal

        for project_dir in dict.fromkeys(d for d in dirs if d):
            shadow = reserved_agent_name_refusal(launch, project_dir)
            if shadow:
                return app_agent_launch_refusal(owner_app, shadow), ""
        return None, launch
    return None, ""


def _release_dispatch_lock(turn) -> None:
    _dispatch_lock = turn._dispatch_lock
    if turn._dispatch_lock_held:
        turn._dispatch_lock_held = False
        _dispatch_lock.release()
