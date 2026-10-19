"""The crew log health counters: labels, buckets, the catalogue gate, accrual.

What each group here locks in:

*   Every attribute value is a closed enum and an unknown one COLLAPSES rather
    than passing through, because a fold name or a cause templated into a series
    is a cardinality bomb the recorder's instrument cache never evicts.
*   The fold catalogue is checked AT IMPORT, in both directions. A fold the crew
    log gained and a fold it lost are different mistakes with different fixes, so
    the failure names both.
*   ``write.lost`` reports a monotonic total accumulated from the deltas it
    reads, so one lost entry is exactly one ``+1``, a reset of the subsystem's
    own counter re-counts nothing, and losses accrued before telemetry was
    enabled are not reported at all.
*   ``fold.apply`` and ``fold.start`` record on the eager fold worker's thread
    and nowhere else. The thread discipline's other half -- that the crew log's
    APPEND path records nothing -- is in ``test_crew_log_health_threads.py``,
    which drives the real paths.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager

import pytest
from opentelemetry.metrics import CallbackOptions

from kiro_crew.metrics import crew_log_health as health
from kiro_crew.metrics.recorder import MetricsRecorder


class _Rec:
    """Recorder double capturing counter calls."""

    def __init__(self, explode: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.explode = explode

    def counter(self, name, value=1, *, unit="1", attrs=None, **_kw):
        if self.explode:
            raise RuntimeError("recorder down")
        assert value == 1, "a crew log health counter only ever adds one"
        self.calls.append((name, dict(attrs or {})))

    def attrs_for(self, name: str) -> "list[dict]":
        return [a for n, a in self.calls if n == name]


@pytest.fixture()
def rec(monkeypatch):
    r = _Rec()
    monkeypatch.setattr("kiro_crew.metrics.provider.get_recorder", lambda: r)
    return r


@pytest.fixture(autouse=True)
def _forget_lost_baselines():
    """The ``write.lost`` baselines are process-wide; one case's are not another's."""
    health.reset_for_tests()
    yield
    health.reset_for_tests()


@contextmanager
def as_fold_worker():
    """Run the body as if on the eager fold worker's thread.

    The claim is released afterwards, not merely left: a claim surviving the body
    would let a later assertion record a fold that production would not.
    """
    health.claim_fold_worker_thread()
    try:
        yield
    finally:
        health.release_fold_worker_thread()


# --------------------------------------------------------------------------- #
# Labels and buckets
# --------------------------------------------------------------------------- #


class TestLabels:
    def test_known_values_pass_through(self):
        assert health.cause_label("overflowed") == "overflowed"
        assert health.scope_label("session") == "session"
        assert health.origin_label("savepoint") == "savepoint"
        assert health.fold_label("radar") == "radar"

    def test_unknown_values_collapse_rather_than_pass_through(self):
        """An unbounded value would be a series the instrument cache never evicts."""
        assert health.cause_label("disk-on-fire") == "other"
        assert health.scope_label("chat-2915") == "other"
        assert health.origin_label("/home/someone/kirocrew") == "other"
        assert health.fold_label("acp-session-41") == "other"

    def test_cardinality_stays_a_small_constant(self):
        apply_series = len(health.SCOPES) * len(health.FOLD_LABELS) * len(health.BUCKETS)
        start_series = len(health.SCOPES) * len(health.ORIGINS)
        assert apply_series + start_series + len(health.CAUSES) <= 256

    def test_every_declared_label_is_its_own_collapse_target(self):
        """A label that collapsed to ``other`` would be a series nothing can reach."""
        for cause in health.CAUSES:
            assert health.cause_label(cause) == cause
        for scope in health.SCOPES:
            assert health.scope_label(scope) == scope
        for origin in health.ORIGINS:
            assert health.origin_label(origin) == origin
        for fold in health.FOLD_LABELS:
            assert health.fold_label(fold) == fold


class TestDurationBuckets:
    @pytest.mark.parametrize(
        "elapsed_ms,expected",
        [
            (-1.0, "lt_10ms"),
            (0.0, "lt_10ms"),
            (9.999, "lt_10ms"),
            (10.0, "lt_100ms"),
            (99.999, "lt_100ms"),
            (100.0, "lt_1s"),
            (999.999, "lt_1s"),
            (1000.0, "ge_1s"),
            (60_000.0, "ge_1s"),
        ],
    )
    def test_each_edge_is_exclusive_at_the_bound(self, elapsed_ms, expected):
        """Exactly 10ms is ``lt_100ms``: the bound belongs to the bucket above it."""
        assert health.duration_bucket(elapsed_ms) == expected

    def test_the_labels_and_the_bounds_describe_one_ladder(self):
        """A bound with no label (or the reverse) would silently mis-bucket."""
        labelled = [label for _bound, label in health._BUCKET_BOUNDS]
        assert labelled + [health.BUCKET_GE_1S] == list(health.BUCKETS)
        assert [bound for bound, _label in health._BUCKET_BOUNDS] == sorted(
            bound for bound, _label in health._BUCKET_BOUNDS
        )


# --------------------------------------------------------------------------- #
# The fold catalogue, checked at import
# --------------------------------------------------------------------------- #


class TestFoldCatalogue:
    """The ``fold`` label set has ONE owner, which is what keeps it honest.

    A second literal copy here would need a check that the two agree, and a
    check that RAISED would be reached through the fold surface's module-scope
    import of this module -- so a forgotten label edit would take down every
    dashboard read and session fold instead of costing a metric. Reading the
    crew log's own catalogue removes the drift rather than detecting it.
    """

    def test_the_label_set_is_the_crew_logs_own_catalogue(self):
        from kiro_crew.crew_log.projection_catalogue import FOLD_NAMES

        assert health.FOLD_LABELS == frozenset(FOLD_NAMES)

    def test_a_fold_the_crew_log_gains_is_labelled_with_no_edit_here(self):
        """No copy to update, so a new fold cannot land silently in ``other``."""
        assert health.fold_label("worktree") == "worktree"
        for name in health.FOLD_LABELS:
            assert health.fold_label(name) != health.OTHER

    def test_a_name_that_is_not_a_fold_collapses(self):
        """``other`` is reached only by a caller passing a non-fold, which is a bug."""
        assert health.fold_label("heartbeat") == "other"
        assert health.fold_label("") == "other"

    def test_importing_this_module_cannot_raise_over_the_catalogue(self):
        """The fold surface imports it at module scope, so import must be total.

        Executed as a module of its own under a catalogue that disagrees with
        anything this module might have believed: it must still load. A fresh
        copy rather than ``importlib.reload`` so a failure could not leave the
        real module half-built for every later case.
        """
        monkey = pytest.MonkeyPatch()
        try:
            # Through the FACADE, never on the owner module. More than one member
            # of the mirrored family binds this name, and the facade's
            # ``__setattr__`` writes through to every holder while its undo
            # restores each one -- patching the owner directly would leave the
            # undo restoring every holder to the patched value.
            monkey.setattr(
                "kiro_crew.crew_log.projection.FOLD_NAMES",
                ("status", "heartbeat"),
            )
            spec = importlib.util.spec_from_file_location("_crew_log_health_probe", health.__file__)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # must not raise
            assert module.FOLD_LABELS == frozenset({"status", "heartbeat"})
        finally:
            monkey.undo()


# --------------------------------------------------------------------------- #
# fold.start / fold.apply
# --------------------------------------------------------------------------- #


def _one_pass(scope: str, origin: str, folds: "tuple[str, ...]" = ()) -> None:
    """One whole fold pass, the way the eager worker drives it.

    Stamp then close, because that IS the only path: a fold branch leaves a mark
    and the worker turns it into points. There is no record-a-start entry point
    to call instead, which is what keeps the fold surface free of a telemetry
    call inside its locks.
    """
    started = health.begin_fold_pass()
    health.note_fold_origin(scope, origin)
    health.note_fold_pass(scope, folds, started)


class TestFoldStart:
    @pytest.mark.parametrize("scope", ["slot", "session"])
    @pytest.mark.parametrize("origin", ["warm", "savepoint", "cold"])
    def test_one_pass_is_one_point_carrying_its_scope_and_origin(self, rec, scope, origin):
        with as_fold_worker():
            _one_pass(scope, origin)
        assert rec.attrs_for(health.FOLD_START) == [{"scope": scope, "origin": origin}]

    def test_nothing_is_recorded_off_the_fold_worker_thread(self, rec):
        """The same branches are reached by a dashboard read, which is not folding."""
        _one_pass("slot", "cold")
        assert rec.calls == []

    def test_an_out_of_enum_value_is_collapsed_at_the_record_site(self, rec):
        with as_fold_worker():
            _one_pass("/home/someone/kirocrew", "acp-first")
        assert rec.attrs_for(health.FOLD_START) == [{"scope": "other", "origin": "other"}]

    def test_a_broken_recorder_costs_the_metric_and_not_the_fold(self, monkeypatch):
        monkeypatch.setattr("kiro_crew.metrics.provider.get_recorder", lambda: _Rec(explode=True))
        with as_fold_worker():
            _one_pass("slot", "cold", ("work",))  # must not raise


class TestFoldPassStamps:
    """The marks a fold branch leaves, and the points the worker makes of them."""

    def test_a_pass_that_stamped_an_origin_records_a_start_and_its_applies(self, rec):
        with as_fold_worker():
            started = health.begin_fold_pass()
            health.note_fold_origin("slot", "cold")
            health.note_fold_pass("slot", ("work",), started)
        assert rec.attrs_for(health.FOLD_START) == [{"scope": "slot", "origin": "cold"}]
        assert len(rec.attrs_for(health.FOLD_APPLY)) == 1

    def test_a_pass_marked_idle_records_its_start_and_no_apply(self, rec):
        """An unchanged live cell answered the pass, so it advanced nothing."""
        with as_fold_worker():
            started = health.begin_fold_pass()
            health.note_fold_origin("slot", "warm")
            health.note_fold_idle("slot")
            health.note_fold_pass("slot", ("work",), started)
        assert rec.attrs_for(health.FOLD_START) == [{"scope": "slot", "origin": "warm"}]
        assert rec.attrs_for(health.FOLD_APPLY) == [], "nothing advanced, nothing applied"

    def test_a_pass_that_stamped_nothing_records_nothing(self, rec):
        """A fold that raised before choosing a start had no pass to describe."""
        with as_fold_worker():
            started = health.begin_fold_pass()
            health.note_fold_pass("slot", ("work",), started)
        assert rec.calls == []

    def test_opening_a_pass_clears_what_the_last_one_left(self, rec):
        """A stamp that survived would be reported by the next, unrelated pass."""
        with as_fold_worker():
            health.begin_fold_pass()
            health.note_fold_origin("slot", "cold")
            health.note_fold_idle("slot")
            started = health.begin_fold_pass()  # a new pass, nothing stamped yet
            health.note_fold_pass("slot", ("work",), started)
        assert rec.calls == []

    def test_a_nested_session_fold_cannot_be_reported_as_the_slot_pass(self, rec):
        """Resolving a slot's units can fold a session log; the marks are keyed."""
        with as_fold_worker():
            started = health.begin_fold_pass()
            health.note_fold_origin("slot", "cold")
            health.note_fold_origin("session", "warm")  # the nested fold
            health.note_fold_pass("slot", ("work",), started)
        assert rec.attrs_for(health.FOLD_START) == [{"scope": "slot", "origin": "cold"}]

    def test_a_pass_is_taken_once_so_a_later_pass_cannot_repeat_it(self, rec):
        with as_fold_worker():
            started = health.begin_fold_pass()
            health.note_fold_origin("session", "savepoint")
            health.note_fold_pass("session", (), started)
            health.note_fold_pass("session", (), started)
        assert len(rec.attrs_for(health.FOLD_START)) == 1

    def test_an_untimed_pass_records_its_start_and_no_duration(self, rec):
        """``begin_fold_pass`` answering ``None`` must not bucket every pass ge_1s."""
        with as_fold_worker():
            health.begin_fold_pass()
            health.note_fold_origin("slot", "cold")
            health.note_fold_pass("slot", ("work",), None)
        assert rec.attrs_for(health.FOLD_START) == [{"scope": "slot", "origin": "cold"}]
        assert rec.attrs_for(health.FOLD_APPLY) == []

    def test_the_stamps_are_a_no_op_off_the_fold_worker_thread(self, rec):
        started = health.begin_fold_pass()
        health.note_fold_origin("slot", "cold")
        health.note_fold_idle("slot")
        health.note_fold_pass("slot", ("work",), started)
        assert rec.calls == []


class TestFoldApply:
    def test_one_applied_fold_is_one_point_with_its_scope_fold_and_bucket(self, rec, monkeypatch):
        ticks = iter([100.0, 100.05])
        monkeypatch.setattr(health, "_now", lambda: next(ticks))
        with as_fold_worker():
            health.note_fold_apply("session", "status", health.begin_fold_pass())
        assert rec.attrs_for(health.FOLD_APPLY) == [
            {"scope": "session", "fold": "status", "bucket": "lt_100ms"}
        ]

    @pytest.mark.parametrize(
        "elapsed_s,expected",
        [
            (0.0, "lt_10ms"),
            (0.009_999, "lt_10ms"),
            (0.010, "lt_100ms"),
            (0.099_9, "lt_100ms"),
            (0.100, "lt_1s"),
            (0.999, "lt_1s"),
            (1.000, "ge_1s"),
            (3.0, "ge_1s"),
        ],
    )
    def test_each_bucket_edge_through_an_injected_clock(
        self, rec, monkeypatch, elapsed_s, expected
    ):
        """An INJECTED clock, never a sleep: the 1s edge is not sleepable in a test.

        The clock starts at zero so the subtraction is exact. A base in the
        thousands (what ``time.monotonic`` actually returns) loses the low bits of
        a 10ms difference, which would make an edge case about float precision
        rather than about the ladder -- the edges themselves are pinned exactly in
        :class:`TestDurationBuckets`.
        """
        ticks = iter([0.0, elapsed_s])
        monkeypatch.setattr(health, "_now", lambda: next(ticks))
        with as_fold_worker():
            health.note_fold_apply("slot", "radar", health.begin_fold_pass())
        assert rec.attrs_for(health.FOLD_APPLY)[0]["bucket"] == expected

    def test_nothing_is_recorded_off_the_fold_worker_thread(self, rec):
        health.note_fold_apply("slot", "radar", health.begin_fold_pass())
        assert rec.calls == []

    def test_an_out_of_enum_value_is_collapsed_at_the_record_site(self, rec, monkeypatch):
        """The record site is the only gate: a caller's value never reaches attrs.

        Without the ``*_label`` calls inside ``note_fold_apply`` this passes a
        session id straight into a series. Testing the label functions alone does
        not catch that, so the value is pushed through the real record site here.

        The clock is PINNED even though this case is about the labels: left on
        the real ``time.monotonic``, a scheduling pause of 10ms between the two
        reads moves the bucket and fails the assertion on correct code.
        """
        monkeypatch.setattr(health, "_now", lambda: 0.0)
        with as_fold_worker():
            health.note_fold_apply("chat-2915", "acp-session-41", health.begin_fold_pass())
        assert rec.attrs_for(health.FOLD_APPLY) == [
            {"scope": "other", "fold": "other", "bucket": "lt_10ms"}
        ]

    def test_a_clock_that_stepped_backwards_still_counts_the_fold(self, rec, monkeypatch):
        ticks = iter([500.0, 499.0])
        monkeypatch.setattr(health, "_now", lambda: next(ticks))
        with as_fold_worker():
            health.note_fold_apply("slot", "work", health.begin_fold_pass())
        assert rec.attrs_for(health.FOLD_APPLY) == [
            {"scope": "slot", "fold": "work", "bucket": "lt_10ms"}
        ]


class TestTelemetryOff:
    def test_nothing_is_recorded_and_no_instrument_is_built(self, monkeypatch):
        """``MetricsRecorder(None)`` is the gate every call site goes through."""
        off = MetricsRecorder(None)
        monkeypatch.setattr("kiro_crew.metrics.provider.get_recorder", lambda: off)
        with as_fold_worker():
            _one_pass("session", "cold", ("status",))
        assert off.enabled is False
        assert off._counters == {}


# --------------------------------------------------------------------------- #
# write.lost
# --------------------------------------------------------------------------- #


@pytest.fixture()
def raw(monkeypatch):
    """A mutable stand-in for the three counts the subsystems keep."""
    counts = {cause: 0 for cause in health.CAUSES}
    monkeypatch.setattr(health, "_read_raw_losses", lambda: dict(counts))
    return counts


class TestWriteLostAccrual:
    def test_the_first_collection_is_a_baseline_not_a_report(self, raw):
        """Losses accrued before telemetry was enabled are not this metric's."""
        raw.update({"dropped": 7, "overflowed": 3, "wake_dropped": 11})
        assert health.write_lost_totals() == {"dropped": 0, "overflowed": 0, "wake_dropped": 0}

    @pytest.mark.parametrize("cause", ["dropped", "overflowed", "wake_dropped"])
    def test_one_loss_is_exactly_one_plus_one_for_its_own_cause(self, raw, cause):
        health.write_lost_totals()
        raw[cause] += 1
        totals = health.write_lost_totals()
        assert totals[cause] == 1
        assert sum(totals.values()) == 1, "a loss must not be counted under another cause"

    def test_a_delta_across_two_collections_is_counted_once(self, raw):
        health.write_lost_totals()
        raw["dropped"] += 4
        assert health.write_lost_totals()["dropped"] == 4
        assert health.write_lost_totals()["dropped"] == 4, "a quiet cycle adds nothing"
        raw["dropped"] += 2
        assert health.write_lost_totals()["dropped"] == 6

    def test_a_counter_replaced_by_a_fresh_one_counts_only_the_fresh_losses(self, raw):
        """A writer rebuilt mid-process starts its own count again at zero."""
        health.write_lost_totals()
        raw["overflowed"] = 5
        assert health.write_lost_totals()["overflowed"] == 5
        raw["overflowed"] = 0  # the subsystem was replaced
        assert health.write_lost_totals()["overflowed"] == 5, "nothing already counted again"
        raw["overflowed"] = 2  # two fresh losses after the replacement
        assert health.write_lost_totals()["overflowed"] == 7

    def test_a_reading_that_is_corrected_downwards_adds_nothing(self, raw):
        """A reading that goes down is harmless whatever the source does.

        ``CrewLogWriter._drop`` never publishes a count it is about to correct,
        so this rule is defence in depth for any source that does: only growth
        is accrued, so a corrected reading cannot become a permanent inflated
        total. Accruing the whole reading on a dip instead reports 41 for the
        sequence below, where the true answer is 1.
        """
        raw["dropped"] = 40
        health.write_lost_totals()  # baseline at 40
        raw["dropped"] = 41  # one real loss
        assert health.write_lost_totals()["dropped"] == 1
        raw["dropped"] = 40  # corrected back down
        assert health.write_lost_totals()["dropped"] == 1
        raw["dropped"] = 42  # two more above the corrected reading
        assert health.write_lost_totals()["dropped"] == 3

    def test_the_total_never_goes_backwards_and_counts_each_rise_once(self, raw):
        """Monotonic, and the arithmetic is pinned -- not just the ordering.

        A test that only asserted ``seen == sorted(seen)`` would pass for an
        over-counting implementation too, so the exact totals are spelled out.
        """
        health.write_lost_totals()
        seen = []
        for reading in (3, 9, 0, 1, 1, 4):
            raw["dropped"] = reading
            seen.append(health.write_lost_totals()["dropped"])
        # +3, +6, replaced, +1, quiet, +3
        assert seen == [3, 9, 9, 10, 10, 13]

    def test_two_readers_collecting_at_once_count_each_loss_once(self, monkeypatch):
        """Two metric readers mean two ticker threads, so collections overlap.

        The read and the accrual have to happen under ONE lock hold. With the
        read outside it, a thread carrying a STALE reading in behind a fresher
        one looks like a counter that went down and then up again, and the same
        loss is added twice.
        """
        import threading

        readings = [5, 7, 7, 7]
        gate = threading.Barrier(2)
        handed: "list[int]" = []

        def slow_read():
            # Both threads enter the read; the one that gets here first is held
            # until the second arrives, which is the overlap being reproduced.
            nth = len(handed)
            handed.append(nth)
            try:
                gate.wait(timeout=5)
            except threading.BrokenBarrierError:  # pragma: no cover - timing floor
                pass
            return {
                "dropped": readings[min(nth, len(readings) - 1)],
                "overflowed": 0,
                "wake_dropped": 0,
            }

        raw_first = {"dropped": 3, "overflowed": 0, "wake_dropped": 0}
        monkeypatch.setattr(health, "_read_raw_losses", lambda: dict(raw_first))
        health.write_lost_totals()  # baseline at 3
        monkeypatch.setattr(health, "_read_raw_losses", slow_read)
        totals: "list[int]" = []

        def collect():
            totals.append(health.write_lost_totals()["dropped"])

        threads = [threading.Thread(target=collect) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert not any(thread.is_alive() for thread in threads)
        # Four real losses happened (3 -> 7). Neither ordering may report more.
        assert max(totals) == 4, f"a loss was counted twice: {totals}"

    def test_a_restart_reports_nothing_it_already_reported(self, raw):
        """The totals live in memory only, so a fresh process starts again at zero."""
        health.write_lost_totals()
        raw["dropped"] = 6
        assert health.write_lost_totals()["dropped"] == 6
        health.reset_for_tests()  # stands in for a fresh process
        assert health.write_lost_totals()["dropped"] == 0


class TestWriteLostSources:
    def test_each_cause_reads_the_count_its_own_subsystem_keeps(self, monkeypatch):
        from kiro_crew.crew_log import eager, emit

        monkeypatch.setattr(emit, "dropped_writes", lambda: 2)
        monkeypatch.setattr(emit, "overflow_writes", lambda *a, **k: 3)
        monkeypatch.setattr(eager, "dropped_wakes", lambda: 4)
        assert health._read_raw_losses() == {
            "dropped": 2,
            "overflowed": 3,
            "wake_dropped": 4,
        }

    def test_a_subsystem_that_was_never_imported_reports_zero(self, monkeypatch):
        """Read through ``sys.modules``: a telemetry callback imports nothing."""
        monkeypatch.delitem(sys.modules, "kiro_crew.crew_log.emit", raising=False)
        monkeypatch.delitem(sys.modules, "kiro_crew.crew_log.eager", raising=False)
        assert health._read_raw_losses() == {cause: 0 for cause in health.CAUSES}

    def test_the_observations_carry_one_point_per_cause(self, raw):
        raw.update({"dropped": 1, "overflowed": 1, "wake_dropped": 1})
        health.write_lost_totals()  # baseline
        raw["dropped"] += 1
        points = {
            (obs.attributes or {})["cause"]: obs.value
            for obs in health._write_lost_observations(CallbackOptions())
        }
        assert points == {"dropped": 1, "overflowed": 0, "wake_dropped": 0}

    def test_a_failing_read_yields_no_observation_rather_than_raising(self, monkeypatch):
        """A callback that raises is dropped by the SDK with the whole collection."""

        def boom():
            raise RuntimeError("writer gone")

        monkeypatch.setattr(health, "_read_raw_losses", boom)
        assert list(health._write_lost_observations(CallbackOptions())) == []
