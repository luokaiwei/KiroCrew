"""A cron mutation whose SAVE fails must not survive in the in-memory job list.

``CronService`` mutators edit ``self._jobs`` in place and only then call
``_save()``. The unreadable-store LATCH (``_load_failed`` ->
``CronStoreUnreadable``) is refused up front by ``_sync_for_write`` and covered
by ``test_cron_store_unreadable_boundaries.py``. A DIFFERENT failure is not: a
save that gets PAST that latch and then raises a bare ``OSError`` from the
atomic write itself -- ``ENOSPC`` (disk full), ``EROFS`` (read-only remount),
``EIO``, or a fault in the ``mkdir`` / ``encode_store`` / ``atomic_write``
chain. On that path the in-memory mutation has already landed, the caller is
told the write failed, yet the change stays in ``self._jobs`` -- visible to
``list_jobs``, firable by the timer, and persisted by the next save that
succeeds.

The contract: a mutator whose owned state is not shared with the execution
thread rolls its in-memory change back on ANY save failure, the way the remove
and update cores already do, extended to the add (×2), ack, unack and adopt
cores plus the loop-stall breaker. ``enable_job`` (pause/resume) CANNOT use a
plain field-scoped restore: its ``enabled``/``auto_paused``/``consecutive_failures``
are execution-owned, so a concurrent ``record_failure``/``record_success`` in the
save window races it in opposite directions (reinstating an auto-pause a success
cleared, or reverting one a failure set). It instead uses an EXECUTION-STATE
REVISION FENCE (``CronJob.exec_state_revision``, bumped at every execution write
of the auto-pause fields): ``_enable_job_locked`` restores
``auto_paused``/``consecutive_failures`` only when the revision is unchanged
across the save, ALWAYS restores the execution-immutable ``user_paused``, and
ALWAYS re-derives ``enabled`` from the settled pause flags. These tests inject the
failure at the ``atomic_write`` boundary (a real ``OSError``, NOT
``CronStoreUnreadable``), so the real ``_save`` -> raise chain runs and the
``_load_failed`` latch is deliberately NOT what is being exercised.

The vacuous-pass trap: asserting only that the mutator RAISES passes whether or
not the rollback exists, because ``_save`` already raised. Every assertion below
is about ``self._jobs`` (and the live object's fields) AFTER the failed save,
and a reload from the untouched file -- the state a timer or a later save would
see.
"""

from __future__ import annotations

import contextlib
import errno
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiro_crew.cron import CronService

# One xdist worker for the whole file, matching the sibling cron suites. One
# group PER FILE, never shared: a shared group would serialize unrelated modules.
pytestmark = pytest.mark.xdist_group(name="tree_scan_test_cron_save_failure_rollback")


@contextlib.contextmanager
def _disk_full_on_save() -> Iterator[None]:
    """Make the next ``atomic_write`` raise ENOSPC, like a full disk would.

    Injected at ``kiro_crew.atomic_write.atomic_write`` -- the exact call
    ``_save`` makes AFTER its ``_load_failed`` guard and its ``mkdir`` -- so the
    real ``_save`` runs, the in-memory mutation has already landed, and a bare
    ``OSError`` (NOT ``CronStoreUnreadable``) comes back out. That distinction is
    the whole point: the latch path is already guarded and tested elsewhere;
    this is the write-fault path the issue names.
    """
    import kiro_crew.atomic_write as aw

    real = aw.atomic_write

    def failing(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    aw.atomic_write = failing  # type: ignore[assignment]
    try:
        yield
    finally:
        aw.atomic_write = real  # type: ignore[assignment]


@contextlib.contextmanager
def _save_fails_after(on_save: "object") -> Iterator[None]:
    """Run ``on_save()`` and THEN fail the save, simulating a concurrent run.

    The execution-state fence exists because ``record_failure`` /
    ``record_success`` run on the loop thread and can write the pause fields in
    the exact window between ``_enable_job_locked``'s mutation and its failing
    ``_save``. This reproduces that window deterministically: the patched
    ``atomic_write`` invokes the callback (which performs the concurrent write
    on the same job object) and only then raises ``ENOSPC``, so the real
    ``_save`` -> raise chain runs with the fields already changed under it.
    """
    import kiro_crew.atomic_write as aw

    real = aw.atomic_write

    def failing(*_args: object, **_kwargs: object) -> None:
        on_save()  # type: ignore[operator]
        raise OSError(errno.ENOSPC, "No space left on device")

    aw.atomic_write = failing  # type: ignore[assignment]
    try:
        yield
    finally:
        aw.atomic_write = real  # type: ignore[assignment]


def _reloaded(tmp_path: Path) -> CronService:
    """A fresh service reading the store from disk -- the timer's/next-save's view."""
    return CronService(base_dir=tmp_path)


# ── add_job ──


def test_a_failed_save_does_not_leave_an_added_job_in_memory(tmp_path: Path) -> None:
    svc = CronService(base_dir=tmp_path)
    assert not svc._load_failed, "precondition: a fresh store loads cleanly"

    with _disk_full_on_save():
        with pytest.raises(OSError) as exc:
            svc.add_job(name="rejected", message="m", every_secs=3600)
    assert exc.value.errno == errno.ENOSPC, "the write fault, not the unreadable latch"

    assert [j.name for j in svc._jobs] == [], (
        "a job whose write failed must not stay in the live list; the caller was "
        f"told the write failed but the scheduler can still see {[j.name for j in svc._jobs]}"
    )
    assert (
        _reloaded(tmp_path).list_jobs(include_disabled=True) == []
    ), "and nothing must be on disk for a later successful save to persist"


def test_a_failed_save_does_not_leave_an_add_if_absent_job_in_memory(tmp_path: Path) -> None:
    svc = CronService(base_dir=tmp_path)

    with _disk_full_on_save():
        with pytest.raises(OSError):
            svc.add_job_if_absent(
                lambda existing: existing.name == "rejected-2",
                name="rejected-2",
                message="m",
                every_secs=3600,
            )

    assert [j.name for j in svc._jobs] == [], "add_job_if_absent left the job in memory"


# ── ack / unack ──


def test_a_failed_save_does_not_record_an_ack_in_memory(tmp_path: Path) -> None:
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="ackable", message="m", every_secs=3600)
    assert svc._jobs[0].acked_items == [], "precondition: nothing acked yet"

    with _disk_full_on_save():
        with pytest.raises(OSError):
            svc.ack_job(job.id, "summary-text")

    assert svc._jobs[0].acked_items == [], "a refused ack must not be recorded in memory"


def test_a_failed_save_does_not_drop_an_unack_in_memory(tmp_path: Path) -> None:
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="ackable", message="m", every_secs=3600)
    svc.ack_job(job.id, "keep-me")
    assert svc._jobs[0].acked_items == ["keep-me"], "precondition: one ack recorded"

    with _disk_full_on_save():
        with pytest.raises(OSError):
            svc.unack_job(job.id)

    assert svc._jobs[0].acked_items == [
        "keep-me"
    ], "a refused unack must not drop the ack from the live cache"


# ── adopt_job (ownership) ──


def test_a_failed_save_does_not_leave_an_adopted_owner_in_memory(tmp_path: Path) -> None:
    """`adopt_job` repoints a job's `session_key` then saves; a failed save
    must not leave the new owner in the live cache (every ownership decision
    in this process reads it)."""
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="ownable", message="m", every_secs=3600)
    svc.adopt_job(job.id, "dashboard:original-owner")
    assert svc._jobs[0].session_key == "dashboard:original-owner", "precondition"

    with _disk_full_on_save():
        with pytest.raises(OSError):
            svc.adopt_job(job.id, "dashboard:new-owner")

    assert (
        svc._jobs[0].session_key == "dashboard:original-owner"
    ), "a refused adoption must leave the cached owner unchanged"
    reloaded = _reloaded(tmp_path).get_job(job.id)
    assert reloaded is not None and reloaded.session_key == "dashboard:original-owner"


# ── the loop-stall breaker's pause ──


def test_a_failed_save_does_not_leave_a_loop_stall_pause_in_memory(tmp_path: Path) -> None:
    """The breaker pauses the job the previous gateway died running. A save
    that fails with a BARE OSError (not CronStoreUnreadable) must roll the
    pause back AND report the verdict unsettled, so the breaker does not claim
    the dump and sweep the markers for a pause that never persisted -- which
    would make the next boot skip the still-enabled job."""
    from kiro_crew import cron_inflight, stall_attribution

    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="stalled", message="m", every_secs=3600)
    assert svc._jobs[0].enabled and not svc._jobs[0].auto_paused, "precondition"

    dump = tmp_path / "crash-dump.txt"
    dump.write_text("stack", encoding="utf-8")
    marker = cron_inflight.RunningMarker(
        job_id=job.id,
        name="stalled",
        started_at=1_700_000_000.0,
        pid=4242,
        path=tmp_path / "cron-running" / f"{job.id}.run.json",
        pid_domain="host:ns",
        pid_start="abc",
    )
    attribution = stall_attribution.StallAttribution(dump=dump, surface="cron", candidates=[marker])

    with _disk_full_on_save():
        settled, paused = svc._pause_for_loop_stall(attribution)

    assert (settled, paused) == (
        False,
        None,
    ), "a save that failed must leave the verdict unsettled so a later boot retries"
    assert svc._jobs[0].enabled and not svc._jobs[0].auto_paused, (
        "a pause whose save failed must not stay in the live cache -- the breaker "
        "went on to claim the dump and sweep the markers, so a surviving pause would "
        "make the next boot skip a job that was never actually paused on disk"
    )
    reloaded = _reloaded(tmp_path).get_job(job.id)
    assert reloaded is not None and reloaded.enabled and not reloaded.auto_paused


# ── NEGATIVE CONTROL: a healthy save must still mutate on every path ──
#
# A rollback keyed on the wrong thing could revert a SUCCESSFUL write too. These
# prove the mutation lands and survives a reload when the save does not fail.


def test_a_healthy_save_still_mutates_on_every_rolled_back_path(tmp_path: Path) -> None:
    svc = CronService(base_dir=tmp_path)

    added = svc.add_job(name="added", message="m", every_secs=3600)
    absent = svc.add_job_if_absent(
        lambda existing: existing.name == "absent",
        name="absent",
        message="m",
        every_secs=3600,
    )
    assert absent is not None
    svc.enable_job(added.id, enabled=False)
    svc.ack_job(added.id, "ack-1")
    svc.unack_job(added.id)
    svc.enable_job(added.id, enabled=True)
    svc.adopt_job(added.id, "dashboard:owner")

    # Every mutation is in the live cache and on disk after a reload.
    live = {j.name: j for j in svc._jobs}
    assert set(live) == {"added", "absent"}
    assert live["added"].enabled and not live["added"].user_paused
    assert live["added"].acked_items == []
    assert live["added"].session_key == "dashboard:owner"

    reloaded = {j.name: j for j in _reloaded(tmp_path).list_jobs(include_disabled=True)}
    assert set(reloaded) == {"added", "absent"}
    assert reloaded["added"].enabled
    assert reloaded["added"].session_key == "dashboard:owner"


# ── enable_job: the execution-state revision fence ──
#
# enable_job (pause/resume) writes enabled/auto_paused/consecutive_failures,
# the SAME fields record_failure/record_success write on the loop thread. A
# plain field-scoped restore races a concurrent run in both directions, so the
# rollback is gated on CronJob.exec_state_revision: restore only when no run
# bumped it across the save, otherwise keep the run's write.


def test_a_failed_pause_rolls_back_when_no_run_touched_the_fields(tmp_path: Path) -> None:
    """The quiet case: no concurrent run, so the revision is unchanged and the
    pause is restored exactly, like the other rollback cores."""
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="pausable", message="m", every_secs=3600)
    assert svc._jobs[0].enabled and not svc._jobs[0].user_paused, "precondition: enabled"

    with _disk_full_on_save():
        with pytest.raises(OSError):
            svc.enable_job(job.id, enabled=False)

    live = svc._jobs[0]
    assert live.enabled and not live.user_paused, (
        "a refused pause with no concurrent run must leave the job enabled in the "
        "live cache -- the scheduler reads it and the disk copy never changed"
    )
    reloaded = _reloaded(tmp_path).get_job(job.id)
    assert reloaded is not None and reloaded.enabled and not reloaded.user_paused


def test_a_failed_resume_rolls_back_when_no_run_touched_the_fields(tmp_path: Path) -> None:
    """The quiet resume case: a failed re-enable of a user-paused job restores
    the paused state exactly."""
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="resumable", message="m", every_secs=3600)
    svc.enable_job(job.id, enabled=False)
    assert svc._jobs[0].user_paused and not svc._jobs[0].enabled, "precondition: paused"

    with _disk_full_on_save():
        with pytest.raises(OSError):
            svc.enable_job(job.id, enabled=True)

    live = svc._jobs[0]
    assert live.user_paused and not live.enabled, (
        "a refused resume with no concurrent run must leave the job paused in the "
        "live cache -- the disk copy still says paused"
    )
    reloaded = _reloaded(tmp_path).get_job(job.id)
    assert reloaded is not None and reloaded.user_paused and not reloaded.enabled


def test_a_concurrent_auto_pause_survives_a_failed_pause_save(tmp_path: Path) -> None:
    """The race the fence exists for: a run crosses the auto-pause threshold in
    the pause's save window. A fixed restore would REVERT that auto-pause; the
    revision changed, so the fence keeps it. The job stays stopped (it was
    being paused anyway), but now for the durable auto_paused reason the run
    recorded, not a transient in-memory-only user pause the next reload drops."""
    from kiro_crew.cron_service.model import _AUTO_PAUSE_THRESHOLD

    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="racing", message="m", every_secs=3600)
    live = svc._jobs[0]
    # One failure below the threshold: the NEXT record_failure auto-pauses.
    live.consecutive_failures = _AUTO_PAUSE_THRESHOLD - 1

    with _save_fails_after(live.record_failure):
        with pytest.raises(OSError):
            svc.enable_job(job.id, enabled=False)

    assert live.auto_paused, (
        "the concurrent record_failure crossed the threshold and set auto_paused "
        "during the save; a field-scoped restore would have reverted it, stranding "
        "a failing job as merely user-paused in memory and enabled on the next reload"
    )
    assert live.consecutive_failures == _AUTO_PAUSE_THRESHOLD, "the run's count is kept"
    assert not live.enabled, "an auto-paused job is not enabled"


def test_a_concurrent_success_survives_a_failed_resume_save(tmp_path: Path) -> None:
    """The opposite race: a run clears an auto-pause via record_success in the
    resume's save window. A fixed restore would REINSTATE the auto-pause; the
    revision changed, so the fence keeps the clear and re-derives enabled from
    the live pause flags."""
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="recovering", message="m", every_secs=3600)
    live = svc._jobs[0]
    # Put it in the auto-paused state a failing run would have produced.
    live.auto_paused = True
    live.enabled = False
    live.consecutive_failures = 5

    with _save_fails_after(live.record_success):
        with pytest.raises(OSError):
            # The user also tries to resume; its save fails.
            svc.enable_job(job.id, enabled=True)

    assert not live.auto_paused, (
        "the concurrent record_success cleared auto_paused during the save; a "
        "field-scoped restore would have reinstated it, so the recovered job would "
        "read as auto-paused in memory until the next reload"
    )
    assert live.consecutive_failures == 0, "the run's counter reset is kept"
    assert not live.user_paused, "the user's resume (user_paused=False) is the authority"
    assert live.enabled, "enabled is re-derived: neither user- nor auto-paused"


def test_record_failures_early_return_does_not_bump_the_revision(tmp_path: Path) -> None:
    """The fence's negative control: a run that started auto-paused and finds
    the pause lifted under it records NOTHING (record_failure early-returns), so
    it must not bump exec_state_revision -- otherwise it would wrongly invalidate
    a concurrent pause/resume rollback though it wrote no field."""
    svc = CronService(base_dir=tmp_path)
    svc.add_job(name="noop-fail", message="m", every_secs=3600)
    live = svc._jobs[0]
    # The exact early-return condition: started auto-paused, pause since lifted.
    live.auto_paused_at_start = True
    live.auto_paused = False
    before = live.exec_state_revision

    live.record_failure()

    assert live.exec_state_revision == before, (
        "record_failure's early-return path wrote no pause field, so it must not "
        "bump the fence and spuriously defeat a concurrent rollback"
    )
    assert live.consecutive_failures == 0, "precondition: the early return counted nothing"


def test_a_failed_resume_reverts_user_paused_even_when_a_run_bumps_the_revision(
    tmp_path: Path,
) -> None:
    """user_paused is NEVER written by execution, so a failed resume must restore
    it to the disk value REGARDLESS of a concurrent run. The revision gate covers
    only auto_paused/consecutive_failures; letting a changed revision keep the
    resume's user_paused=False would leave the job enabled in memory though the
    disk copy the failed save never touched still says user-paused."""
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="resume-race", message="m", every_secs=3600)
    svc.enable_job(job.id, enabled=False)  # user-paused, saved
    live = svc._jobs[0]
    assert live.user_paused and not live.enabled, "precondition: user-paused on disk"

    # A run recovers (record_success) during the resume's failing save -- it
    # bumps the revision and clears auto-pause state, but touches no user_paused.
    with _save_fails_after(live.record_success):
        with pytest.raises(OSError):
            svc.enable_job(job.id, enabled=True)

    assert live.user_paused, (
        "the resume's save failed, so the disk copy is still user-paused; a "
        "concurrent run never writes user_paused, so it must be restored -- not "
        "left at the resume's False"
    )
    assert not live.enabled, "enabled is re-derived: a user-paused job is not enabled"
    reloaded = _reloaded(tmp_path).get_job(job.id)
    assert reloaded is not None and reloaded.user_paused and not reloaded.enabled


def test_a_concurrent_oneshot_autopause_survives_a_failed_pause_save(tmp_path: Path) -> None:
    """The fence must observe EVERY execution write of the auto-pause fields, not
    only record_failure/record_success. The failed-one-shot auto-pause path bumps
    the revision through the same _note_exec_state_change helper, so a plain
    auto_paused write landing in a pause's save window is kept, not reverted."""
    svc = CronService(base_dir=tmp_path)
    job = svc.add_job(name="oneshot-race", message="m", every_secs=3600)
    live = svc._jobs[0]

    def concurrent_autopause() -> None:
        # Exactly the write run_execution's failed-one-shot branch performs,
        # routed through the same helper the real site uses.
        live.enabled, live.auto_paused = False, True
        live._note_exec_state_change()

    with _save_fails_after(concurrent_autopause):
        with pytest.raises(OSError):
            svc.enable_job(job.id, enabled=False)

    assert live.auto_paused, (
        "an execution auto_paused write that is not record_failure still bumps the "
        "fence, so a field-scoped restore must not revert it"
    )
    assert not live.enabled, "an auto-paused job is not enabled"
