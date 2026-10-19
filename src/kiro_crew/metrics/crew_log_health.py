"""Crew log health counters: is the log losing data, and how is it folded?

Three counters, and all three exist because the crew log is the record every
dashboard, conductor and work ledger reads, and nothing today reports whether
that record is complete or how expensive it is to rebuild:

``kirocrew.crew_log.write.lost``
    Entries the log never got, split by ``cause``. A non-zero reading is a real
    hole: a storage refusal the writer gave up on (``dropped``), an append the
    buffer rejected at its memory ceiling (``overflowed``), or a fold wake the
    eager queue could not hold (``wake_dropped`` -- the folds stay correct, the
    dashboard just sees the value later).
``kirocrew.crew_log.fold.apply``
    One point per fold applied over one batch, with the pass duration as a
    closed ``bucket``. Answers "is folding getting slower, and which fold".
``kirocrew.crew_log.fold.start``
    One point per fold start, by where the state came from: a warm in-memory
    cell, a savepoint on disk, or a cold rebuild from the first record. A rising
    ``cold`` share is the signal that savepoints are not being kept.

THE APPEND THREAD GAINS NOTHING, which is the constraint this module is shaped
by rather than a property it happens to have. The crew log's append path is one
``put_nowait`` plus a set membership test (``eager.note_commit``), and a turn
waits on it, so a counter call there would be new work on the one path that must
stay cheap. Two mechanisms keep it out:

*   Every ``write.lost`` cause is detected ON the append thread or on a
    producer's own thread, and each one is ALREADY counted by the subsystem that
    detects it -- ``WriterStats.dropped`` / ``.overflowed`` for the writer,
    ``eager.dropped_wakes()`` for the fold queue. So this module adds no call at
    those sites at all: it registers ONE observable counter whose callback runs
    on the metrics reader's own ticker thread, reads those three numbers (each an
    O(1) counter read, no disk, no lock held across anything) and reports what
    has accrued since the last collection. ``test_crew_log_health_threads.py``
    pins that the append path records nothing.
*   ``fold.apply`` and ``fold.start`` are recorded only on the eager fold
    worker thread. The worker CLAIMS its thread once at start
    (:func:`claim_fold_worker_thread`), and every record site begins with a
    thread-local lookup returning immediately on every other thread -- so the
    same fold-start branches, reached from a dashboard read or from a CLI,
    record nothing. That is also what keeps the metric a count of EAGER folding
    rather than of reads, which is the question section 6.5 asks.

NOTHING IS RECORDED FROM INSIDE A FOLD, and that is the second half of the same
idea. Two of the start branches run while holding a per-cell fold lock a
dashboard read can be waiting on, and a FIRST ``get_recorder()`` builds the whole
telemetry provider synchronously. So a branch only leaves a thread-local mark
(:func:`note_fold_origin`, :func:`note_fold_idle`), and the eager worker turns the
marks into points afterwards, outside every lock, with :func:`note_fold_pass`.
The marks are keyed by scope because a pass can nest: resolving which units a
slot fold covers can fold a session log.

WHAT ``fold.apply`` MEANS, exactly, because it is narrower than "a fold ran". It
counts a fold THE EAGER WORKER'S OWN PASS ADVANCED. A pass an unchanged live cell
answered advanced nothing and records a start with no apply, as does one that was
refused or that raised -- so ``fold.start`` above ``fold.apply`` is the share of
eager folding that produced no value. One consequence is worth knowing before
reading the ratio: when a dashboard READ reaches a growing cell first, the read
does the folding and the worker's pass behind it finds nothing changed, so that
fold is counted as a start and not as an apply. The two series together say how
much work eager folding is doing, not how much folding the process did.

The duration bucket measures the pass end to end, INCLUDING any wait for the
per-cell fold lock. A pass held up behind a reader is slow eager folding, which
is the thing being measured.

CARDINALITY. Every attribute value is a closed enum: three causes, two scopes,
three origins, four buckets, and the fold catalogue
(:data:`~kiro_crew.crew_log.projection_catalogue.FOLD_NAMES`, 16 names). The
catalogue is read straight from its one owner, so a fold the crew log gains is a
fold this metric labels. Nothing here raises at IMPORT, which matters because the
fold surface imports this module at module scope: a second literal copy of that
catalogue, with a check that raised when the two drifted, would make a forgotten
label edit take down every dashboard read and session fold rather than costing a
metric.

TEMPORALITY, and why ``write.lost`` is a GAUGE carrying a total. ``fold.apply``
and ``fold.start`` are synchronous counters and export DELTA like every other
counter (``metrics/temporality.py``). ``write.lost`` reads a total it does not
own, which can only be an asynchronous instrument -- and an observable COUNTER
would export CUMULATIVE, which this project allows nowhere: one cumulative series
forces every consumer into stateful whole-hour aggregation, and
``test_no_instrument_exports_cumulative`` asserts it over every exported
instrument. Declaring DELTA for it instead is the move ``temporality.py`` records
as tried and reverted, because the delta baseline lives in the provider and the
provider is rebuilt whenever telemetry consent changes, so the first collection
after a rebuild would re-emit a whole process total as one delta. So this follows
the shape the process CPU and GC readings already took for the same reason: an
observable GAUGE whose reading IS the process-lifetime total. A gauge carries no
temporality, merges one datapoint at a time, and the loss RATE is the consumer's
difference between two readings.

The value reported is a total this module accumulates itself from the growth it
reads, so it is monotonic even across a counter that is replaced by a fresh one,
and it starts at zero for the process rather than at whatever the writer had
accrued before telemetry was enabled. :func:`write_lost_totals` reads and accrues
under ONE lock hold because more than one reader collects, and :func:`_accrue`
accrues only growth because a reading can go down for two unrelated reasons; both
are explained there.

Everything is behind ``telemetry.enabled``: ``get_recorder()`` is a no-op
recorder when telemetry is off, and the observable counter is registered only on
``provider._build_recorder()``'s live path.

OSS-CLEAN: stdlib + ``opentelemetry`` (Apache-2.0) + first-party only.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import TYPE_CHECKING, Final, Iterable

from kiro_crew.crew_log.projection_catalogue import FOLD_NAMES

if TYPE_CHECKING:
    from opentelemetry.metrics import CallbackOptions, Meter, Observation

    # The recorder's own attribute type, so the value union has ONE owner. A
    # ``dict[str, str]`` is not assignable to it (``Dict`` is invariant), and
    # restating the union here would be a second copy to drift. Annotation-only:
    # this module is imported by the fold surface at module scope and must not
    # pull the recorder, and through it the OTEL metrics SDK, onto that path.
    from kiro_crew.metrics.recorder import Attributes

logger = logging.getLogger(__name__)

WRITE_LOST: Final[str] = "kirocrew.crew_log.write.lost"
FOLD_APPLY: Final[str] = "kirocrew.crew_log.fold.apply"
FOLD_START: Final[str] = "kirocrew.crew_log.fold.start"

#: Why an entry or a wake was lost. ``dropped`` and ``overflowed`` are holes in
#: the log itself and are different failures (a stuck disk vs a saturated one);
#: ``wake_dropped`` is not a hole -- the next read carries the fold forward -- so
#: it is a separate cause rather than folded into the other two.
CAUSE_DROPPED: Final[str] = "dropped"
CAUSE_OVERFLOWED: Final[str] = "overflowed"
CAUSE_WAKE_DROPPED: Final[str] = "wake_dropped"
CAUSES: Final[tuple[str, ...]] = (CAUSE_DROPPED, CAUSE_OVERFLOWED, CAUSE_WAKE_DROPPED)

#: What a fold is keyed by. A slot fold joins every unit of one board; a session
#: fold reads one unit's file.
SCOPE_SLOT: Final[str] = "slot"
SCOPE_SESSION: Final[str] = "session"
SCOPES: Final[tuple[str, ...]] = (SCOPE_SLOT, SCOPE_SESSION)

#: Where a fold's starting state came from.
ORIGIN_WARM: Final[str] = "warm"
ORIGIN_SAVEPOINT: Final[str] = "savepoint"
ORIGIN_COLD: Final[str] = "cold"
ORIGINS: Final[tuple[str, ...]] = (ORIGIN_WARM, ORIGIN_SAVEPOINT, ORIGIN_COLD)

#: The fold-duration buckets, as a counter attribute rather than a histogram:
#: section 4 of the RFC these metrics come from allows no new histogram, because
#: a backend that does not keep explicit histogram buckets loses the shape. The
#: EDGES ARE STATED HERE AND IN ``docs/system-specs/modules/metrics.md`` and are
#: the only place they are written, so the two cannot drift.
BUCKET_LT_10MS: Final[str] = "lt_10ms"
BUCKET_LT_100MS: Final[str] = "lt_100ms"
BUCKET_LT_1S: Final[str] = "lt_1s"
BUCKET_GE_1S: Final[str] = "ge_1s"
BUCKETS: Final[tuple[str, ...]] = (
    BUCKET_LT_10MS,
    BUCKET_LT_100MS,
    BUCKET_LT_1S,
    BUCKET_GE_1S,
)

#: ``(upper bound in ms, label)``, lowest first, each bound EXCLUSIVE -- so
#: exactly 10.0ms is ``lt_100ms`` and not ``lt_10ms``. Everything at or above the
#: last bound is :data:`BUCKET_GE_1S`.
_BUCKET_BOUNDS: Final[tuple[tuple[float, str], ...]] = (
    (10.0, BUCKET_LT_10MS),
    (100.0, BUCKET_LT_100MS),
    (1000.0, BUCKET_LT_1S),
)

#: The fold names this metric may carry: the crew log's own catalogue, resolved
#: ONCE at import rather than per record. One owner, so there is no second copy
#: to drift -- a fold the crew log gains is a fold this metric labels, and the
#: series count stays bounded because the catalogue is closed (16 names today,
#: listed in ``docs/system-specs/modules/metrics.md`` for a reader).
FOLD_LABELS: Final[frozenset[str]] = frozenset(FOLD_NAMES)

#: Collapse target for a value outside its enum (RFC section 4 rule 3).
OTHER: Final[str] = "other"

_DESCRIPTIONS: Final[dict[str, str]] = {
    WRITE_LOST: "Crew log entries or fold wakes lost, by cause",
    FOLD_APPLY: "Crew log folds applied over one eager batch",
    FOLD_START: "Crew log fold starts, by where the state came from",
}


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def cause_label(cause: str) -> str:
    """Collapse *cause* to a member of :data:`CAUSES`."""
    return cause if cause in CAUSES else OTHER


def scope_label(scope: str) -> str:
    """Collapse *scope* to a member of :data:`SCOPES`."""
    return scope if scope in SCOPES else OTHER


def origin_label(origin: str) -> str:
    """Collapse *origin* to a member of :data:`ORIGINS`."""
    return origin if origin in ORIGINS else OTHER


def fold_label(fold: str) -> str:
    """Collapse *fold* to a member of :data:`FOLD_LABELS`.

    The catalogue is the crew log's own and is closed, so reaching ``other`` here
    means a caller passed something that is not a registered fold at all -- a
    bug, and one worth seeing as a series rather than as a cardinality leak.
    """
    return fold if fold in FOLD_LABELS else OTHER


def duration_bucket(elapsed_ms: float) -> str:
    """Which of :data:`BUCKETS` *elapsed_ms* falls in. Bounds are exclusive.

    A negative reading (a clock that stepped backwards between the two samples)
    lands in the lowest bucket rather than raising: the pass did happen, and the
    count is the signal.
    """
    for bound, label in _BUCKET_BOUNDS:
        if elapsed_ms < bound:
            return label
    return BUCKET_GE_1S


# --------------------------------------------------------------------------- #
# The clock, and the thread that may record a fold
# --------------------------------------------------------------------------- #

#: Indirection so a test can drive the bucket edges with values it chooses.
#: Sleeping to cross an edge would make the test both slow and flaky, and the
#: 1s edge is not reachable by sleeping at all within a sane test budget.
_now = time.monotonic

#: The eager fold worker's claim on its own thread, and the stamps a fold pass
#: leaves on it. A ``threading.local`` rather than the worker's thread object, so
#: this module holds no reference to the crew log's state and the check costs one
#: attribute lookup.
_claim = threading.local()


def claim_fold_worker_thread() -> None:
    """Mark the CALLING thread as the eager fold worker.

    Called once by the worker as it starts. Everything in this module that
    records a fold refuses to record on any other thread, which is what makes
    ``fold.apply`` and ``fold.start`` counts of eager folding rather than of
    every read that happens to fold the same cell.
    """
    _claim.fold_worker = True


def release_fold_worker_thread() -> None:
    """Undo :func:`claim_fold_worker_thread` for the calling thread.

    For a test that borrows its own thread to stand in for the worker: a claim
    left behind would make the next assertion on that thread record a fold that
    production would not.
    """
    _claim.fold_worker = False


def on_fold_worker_thread() -> bool:
    """Whether the calling thread is the one allowed to record a fold."""
    return getattr(_claim, "fold_worker", False) is True


# --------------------------------------------------------------------------- #
# fold.start / fold.apply -- eager worker thread only
# --------------------------------------------------------------------------- #


def begin_fold_pass() -> "float | None":
    """Open a fold pass on this thread: clear its stamps, answer the start instant.

    ``None`` when the clock cannot be read, which :func:`note_fold_pass` treats
    as "do not record a duration" rather than recording a wrong one -- a start
    instant of ``0.0`` would have bucketed every pass as ``ge_1s``.

    The stamps are cleared HERE rather than only when they are taken, because a
    fold that raises before it reaches a start branch leaves whatever the
    previous pass stamped. Clearing at the open means a pass can only ever
    report its own.
    """
    _claim.origin = {}
    _claim.idle = set()
    try:
        return _now()
    except Exception:  # pragma: no cover - a clock that will not read
        logger.debug("crew log fold clock unavailable", exc_info=True)
        return None


def note_fold_origin(scope: str, origin: str) -> None:
    """Stamp where the *scope* pass on this thread got its starting state.

    RECORDS NOTHING. This is called from inside the fold's own branches, and two
    of those hold a per-cell lock: a first ``get_recorder()`` there can build the
    whole telemetry provider synchronously and would do it while holding a lock a
    dashboard read is waiting on. So the branch leaves a thread-local mark and
    :func:`note_fold_pass` turns it into a point afterwards, outside every lock.

    KEYED BY SCOPE because a pass can nest: resolving which units a slot fold
    covers can fold a session log, and an unkeyed mark would let the inner fold's
    origin be reported as the outer one's.

    A NO-OP on every thread but the eager fold worker's, and the check is the
    first thing it does -- so the start branches stay free for a dashboard read,
    a CLI and the append path alike.
    """
    if not on_fold_worker_thread():
        return
    marks = getattr(_claim, "origin", None)
    if marks is None:
        marks = _claim.origin = {}
    # LAST ONE WINS for a given scope. ``fold_session`` folds twice when the file
    # changed identity under the first pass, and the origin that describes what
    # the caller is actually served is the one the retry folded from.
    marks[scope] = origin


def note_fold_idle(scope: str) -> None:
    """Stamp that the *scope* pass on this thread advanced nothing.

    RECORDS NOTHING, for the reason :func:`note_fold_origin` gives. Without this
    a pass that an unchanged live cell answered would count as an apply, and
    ``fold.start`` above ``fold.apply`` would stop meaning "folding that produced
    no value".
    """
    if not on_fold_worker_thread():
        return
    idle = getattr(_claim, "idle", None)
    if idle is None:
        idle = _claim.idle = set()
    idle.add(scope)


def note_fold_pass(scope: str, folds: "Iterable[str]", started: "float | None") -> None:
    """Close the *scope* pass: one ``fold.start``, and one ``fold.apply`` per fold.

    Called by the eager fold worker once per pass, OUTSIDE any fold lock. Takes
    and clears this thread's stamps for *scope*, so a later pass cannot report
    them again.

    No start was stamped means the fold raised before it chose one, and nothing
    is recorded: there was no pass to describe. A pass stamped IDLE advanced
    nothing, so it records its start and no apply.
    """
    if not on_fold_worker_thread():
        return
    marks = getattr(_claim, "origin", None) or {}
    idle = getattr(_claim, "idle", None) or set()
    origin = marks.pop(scope, None)
    was_idle = scope in idle
    idle.discard(scope)
    if origin is None:
        return
    _count(FOLD_START, {"scope": scope_label(scope), "origin": origin_label(origin)})
    if was_idle or started is None:
        return
    for fold in folds:
        note_fold_apply(scope, fold, started)


def note_fold_apply(scope: str, fold: str, started: "float | None") -> None:
    """Count one application of *fold* over one batch, timed from *started*.

    ``None`` for *started* means the pass could not be timed, and then NOTHING is
    recorded: a duration is the whole point of this counter, and inventing one
    from a missing instant would bucket every such pass identically. The
    parameter accepts it rather than demanding a float so that every caller
    holding :func:`begin_fold_pass`'s answer can hand it straight over, with one
    place deciding what an untimed pass means.

    *started* comes from :func:`begin_fold_pass`. Only a fold that APPLIED is
    counted: a pass the read path refused, one that raised, and one that found
    nothing to move all record nothing, so ``fold.start`` above ``fold.apply``
    is how much folding is starting without producing a value.

    A NO-OP on every thread but the eager fold worker's, as :func:`note_fold_origin`
    is and for the same reason.
    """
    if started is None or not on_fold_worker_thread():
        return
    elapsed_ms = (_now() - started) * 1000.0
    _count(
        FOLD_APPLY,
        {
            "scope": scope_label(scope),
            "fold": fold_label(fold),
            "bucket": duration_bucket(elapsed_ms),
        },
    )


def _count(name: str, attrs: "Attributes") -> None:
    """Add one to the counter *name*. Never raises.

    The provider import is function-local: this module is reached from the crew
    log's fold surface, and a module-scope import would put the OpenTelemetry
    metrics SDK's import chain behind it.
    """
    try:
        from kiro_crew.metrics.provider import get_recorder

        get_recorder().counter(name, 1, unit="1", attrs=attrs, description=_DESCRIPTIONS[name])
    except Exception:  # noqa: BLE001 - telemetry must never cost a fold
        logger.debug("crew log health counter %r failed", name, exc_info=True)


# --------------------------------------------------------------------------- #
# write.lost -- the metrics reader thread only
# --------------------------------------------------------------------------- #

#: Guards the two tables below. The callback runs on the reader's ticker thread
#: and a second reader (the local sink plus an OTLP destination) has its own, so
#: two collections can overlap.
_lost_guard = threading.Lock()
#: Per cause, the last raw reading taken from the owning subsystem. ``None``
#: until the first collection, which is the BASELINE: losses a process accrued
#: before telemetry was enabled are not this metric's to report.
_lost_seen: "dict[str, int]" = {}
#: Per cause, the monotonic total this module has accumulated from the deltas it
#: read. In memory only, so a restart starts again at zero and no loss counted
#: before it is counted again.
_lost_total: "dict[str, int]" = {}


def reset_for_tests() -> None:
    """Forget the ``write.lost`` baselines and totals.

    A test seam, named as one. The tables are process-wide, so one case's
    readings would otherwise become the next case's baseline.
    """
    with _lost_guard:
        _lost_seen.clear()
        _lost_total.clear()


def _read_raw_losses() -> "dict[str, int]":
    """The three loss counts their owning subsystems already keep.

    Read through ``sys.modules`` rather than by importing, and that is not a
    micro-optimisation: this runs on the metrics reader's ticker thread, and
    importing the crew log's writer or fold queue FROM a telemetry callback would
    put that whole import chain on a thread no caller is waiting on, in a process
    that may never have appended anything. A module that was never imported has
    nothing to report, which is exactly ``0``.
    """
    counts = {cause: 0 for cause in CAUSES}
    emit = sys.modules.get("kiro_crew.crew_log.emit")
    if emit is not None:
        counts[CAUSE_DROPPED] = int(emit.dropped_writes())
        counts[CAUSE_OVERFLOWED] = int(emit.overflow_writes())
    eager = sys.modules.get("kiro_crew.crew_log.eager")
    if eager is not None:
        counts[CAUSE_WAKE_DROPPED] = int(eager.dropped_wakes())
    return counts


def _accrue(cause: str, current: int) -> int:
    """Fold *current* into *cause*'s monotonic total and return it.

    ONLY GROWTH IS ACCRUED, and a reading lower than the last one moves the
    watermark down without adding anything. That rule has to hold both ways round
    for two different reasons:

    *   A counter can be REPLACED by a fresh one. A writer rebuilt mid-process
        starts again at zero, and lowering the watermark means its next loss is
        counted once rather than its whole new total being added to a total that
        already stands.
    *   A counter can momentarily read HIGH. ``CrewLogWriter._drop`` adds a whole
        failed batch to its dropped count and then takes the follow-on jobs back
        out, so a reader landing between the two sees a count that is about to be
        corrected. Treating that dip as a fresh counter (adding the whole new
        reading) turned one requeued job into an inflated total that never came
        back down -- it is the defect this rule exists to prevent.

    A corrected reading cannot be un-counted, because an observable counter's
    observation must not go backwards; the source is fixed instead, so the
    transient is never published (see ``CrewLogWriter._drop``).
    """
    seen = _lost_seen.get(cause)
    _lost_seen[cause] = current
    if seen is None:
        # First collection: adopt the reading as the baseline. Losses from before
        # telemetry was enabled are not reported -- see the module docstring.
        return _lost_total.setdefault(cause, 0)
    total = _lost_total.get(cause, 0) + max(current - seen, 0)
    _lost_total[cause] = total
    return total


def write_lost_totals() -> "dict[str, int]":
    """Each cause's monotonic total, after folding in the current readings.

    THE READ IS INSIDE THE LOCK, which is the whole correctness of the function
    rather than a precaution. Kiro Crew attaches more than one reader to one
    meter provider -- the local JSONL sink always, plus a reader per OTLP
    destination -- and each has its own ticker thread, so two collections can
    overlap. Reading outside the lock lets the slower thread carry a STALE
    reading into the accrual behind the fresher one, where it reads as a counter
    that went backwards and then forwards again: the same loss is then added
    twice. The read is three O(1) counter reads, so holding the lock across it
    costs nothing, and no path takes this lock while holding the writer's or the
    fold queue's, so it cannot deadlock against them.

    Public so a test can assert the accrual without an OTEL reader, and so the
    callback below is one line over it.
    """
    with _lost_guard:
        raw = _read_raw_losses()
        return {cause: _accrue(cause, raw[cause]) for cause in CAUSES}


def _write_lost_observations(
    options: "CallbackOptions",
) -> "Iterable[Observation]":
    """One observation per cause, for the observable ``write.lost`` gauge."""
    from opentelemetry.metrics import Observation

    try:
        totals = write_lost_totals()
    except Exception:  # noqa: BLE001 - a telemetry callback must never raise
        logger.debug("crew log write.lost read failed", exc_info=True)
        return ()
    return [Observation(total, {"cause": cause_label(cause)}) for cause, total in totals.items()]


def register_crew_log_health(meter: "Meter") -> None:
    """Register the observable ``write.lost`` gauge on *meter*.

    Called from ``provider._build_recorder()``'s live path, so the
    ``telemetry.enabled`` consent gate covers it. The callback runs only when a
    reader collects, on that reader's own ticker thread -- no sampler of ours, no
    work at rest, and nothing on the crew log's append thread.

    THE BASELINES AND TOTALS ARE NOT TOUCHED HERE. Registering is not the start
    of the stream: the recorder is rebuilt in-process whenever telemetry consent
    changes, and ``provider._consent_worker`` can build a recorder it then
    DISCARDS because a newer consent superseded it. A registration that cleared
    this module's state would let that discarded build zero the totals of the
    provider still serving, which both loses the losses already accrued and makes
    the series go backwards. The baseline is taken instead by the first
    COLLECTION of the process, which is what keeps losses accrued before
    telemetry was enabled out of the stream.
    """
    from kiro_crew.metrics.schema import validate_name

    meter.create_observable_gauge(
        name=validate_name(WRITE_LOST),
        unit="1",
        description=_DESCRIPTIONS[WRITE_LOST],
        callbacks=[_write_lost_observations],
    )
