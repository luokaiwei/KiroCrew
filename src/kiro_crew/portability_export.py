"""Building the export zip.

The walk that reads the live home by descriptor and refuses a link at any level, the
SQLite backups taken from a checkpointed database, the session transcripts and their
companions within the manifest's budget, and :func:`create_export_zip`, which writes
the archive and its manifest.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import stat
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePath, PurePosixPath

from kiro_crew import pinned_fs, platform_compat
from kiro_crew._sqlite_compat import sqlite3
from kiro_crew.chat_attachments import ATTACHMENTS_DIR_SUFFIX
from kiro_crew.history import (
    ARCHIVE_DIR_NAME,
    SESSIONS_DIR_NAME,
    THREADS_DIR_NAME,
    THREADS_SIDECAR_SUFFIX,
    ConversationLog,
)
from kiro_crew.member_memory_backup import hold_stores_for_read
from kiro_crew.memory_stores import MEMORY_STORES_DIR_NAME, is_host_local_store_state
from kiro_crew.portability_archive import (
    _MAX_IMPORT_MEMBERS,
    _MAX_IMPORT_UNCOMPRESSED,
    _MAX_SETTINGS_DOCUMENT_BYTES,
    _SETTINGS_DOCUMENTS,
    _TRANSCRIPT_SUFFIX,
    MEMORY_EXPORT_MANIFEST_VERSION,
    MEMORY_ONLY_TREES,
    _chat_stem,
    _entry_is_link,
    _is_excluded,
    _is_rotation_header,
    _keep_top_level_transcript,
    _mc_dir,
    _read_first_line_fd,
    _rotation_segment_for_stem,
    _transcript_header_verdict,
    logger,
)
from kiro_crew.portability_templates import (
    AGENT_TEMPLATES_DIR,
    bundled_agent_templates,
)
from kiro_crew.security import is_sensitive_path
from kiro_crew.snapshot import _DB_SIDECAR_GLOBS, EXPORT_MANIFEST_VERSION, is_product_tree_database


def _wal_checkpoint(db_path: Path) -> None:
    if db_path.is_file():
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
        except Exception:
            logger.debug("WAL checkpoint failed for %s", db_path)


def _backup_sqlite(src: Path, dst_buffer: io.BytesIO) -> None:
    """Use SQLite backup API for a consistent copy."""
    src_conn = sqlite3.connect(src.absolute().as_uri() + "?mode=ro", uri=True)
    mem_conn = sqlite3.connect(":memory:")
    try:
        src_conn.backup(mem_conn)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        try:
            disk_conn = sqlite3.connect(tmp.name)
            try:
                mem_conn.backup(disk_conn)
            finally:
                disk_conn.close()
            dst_buffer.write(Path(tmp.name).read_bytes())
        finally:
            os.unlink(tmp.name)
    finally:
        src_conn.close()
        mem_conn.close()


def _keep_for_export(rel: PurePath) -> bool:
    """The old loop's by-name filters, unchanged and in the same order.

    Purely lexical, and it has to stay that way: this runs on a child the walk
    has not verified yet, so a filesystem call here would be the very probe the
    walk exists to remove.

    The old loop also asked ``is_sensitive_path`` at this point, about the
    PATHNAME. That call is gone rather than moved, because resolving an
    unverified name is the probe. The same question is still asked -- in
    :func:`_open_verified`, of the descriptor's real path -- and that was always
    the load-bearing one: a name can be re-pointed between the check and the
    open, a descriptor cannot.
    """
    # ``PurePosixPath(*rel.parts)``, never ``PurePosixPath(str(rel))``: on Windows
    # ``str(rel)`` is backslash-separated, so ``PurePosixPath`` parses the whole
    # relative path as a SINGLE component. ``.name`` is then the entire path and
    # ``.parts`` has length one, so the ``EXPORT_EXCLUDE`` basename set and the
    # ``EXCLUDE_DIRS`` walk both stop matching -- ``workspace/notes/.env`` was
    # exported on Windows. Rebuilding from ``parts`` keeps the separator the
    # exclusion rules are written against.
    if _is_excluded(PurePosixPath(*rel.parts)):
        return False
    return not (rel.parts[0] == "skills" and "auto" in rel.parts)


def _keep_store_for_export(rel: PurePath) -> bool:
    """`_keep_for_export` for the ``memory_stores/`` tree, plus that tree's own two rules.

    Host-local state -- the member signing key, execution logs, local backup directories --
    stays on this host (`memory_stores.is_host_local_store_state`, the same predicate the
    snapshot applies). SQLite sidecars stay too: a store's databases are copied through the
    backup API below, which yields a self-contained file, and a ``-wal`` archived next to
    that copy would be replayed into a database it never belonged to. Lexical only, like
    the filter it extends.
    """
    if not _keep_for_export(rel):
        return False
    posix = PurePosixPath(*rel.parts)
    if is_host_local_store_state(posix.parts):
        return False
    return not any(posix.match(glob) for glob in _DB_SIDECAR_GLOBS)


def _walk_contained(
    root_real: str,
    rel_dir: PurePath,
    keep: Callable[[PurePath], bool],
    *,
    fenced_ok: bool = False,
) -> Iterator[tuple[PurePath, int]]:
    """Yield ``(rel, fd)`` for every regular file under ``root_real / rel_dir``.

    The export's own enumeration, in place of ``rglob``. ``rglob`` DESCENDS a
    Windows junction, and every by-name question then asked about what it yields
    resolves that junction before any guard has run. Measured on this module
    before the change, exporting a workspace holding one pre-planted junction:
    twelve resolving calls went out through it -- four ``os.path.realpath``,
    eight ``ntpath._getfinalpathname`` -- while the export correctly archived
    nothing from behind it. Nothing being archived was never the question. If the
    junction names a UNC share those resolutions are the outbound authentication,
    and containment refusing the bytes afterwards has already paid the cost it
    exists to prevent.

    So this descends and verifies in one motion, root first:

    1. the directory is PINNED. :func:`platform_compat.pin_directory` opens it
       with ``OPEN_REPARSE_POINT``, so a junction sitting at the name fails here
       instead of being traversed -- the refusal and the open are one operation,
       not a check followed by an open. On Windows the handle also omits
       ``FILE_SHARE_DELETE``, so while it lives neither that directory nor
       anything above it can be renamed or deleted;
    2. only then is it listed, and each child classified from the listing itself
       (:func:`_entry_is_link`) -- nothing resolves a child;
    3. a child directory is descended only by re-entering at step 1, so the path
       the kernel walks to reach component *n* runs entirely through components
       already opened and verified;
    4. a child file is opened by :func:`_open_verified`, which does not follow a
       reparse point at the final name, while its whole parent chain is still
       held.

    Each pin is held for as long as the subtree under it is being produced --
    a generator frame stays alive across ``yield``, so an outer level's handle
    outlives the inner walk -- and released in ``finally``, which also runs when
    the consumer abandons the walk. Depth, not breadth, bounds how many are open.

    *keep* is asked about relative paths only and must not touch the filesystem.
    It is applied to FILES only, exactly where the old loop applied it: matching
    a directory NAME does not prune its contents, because ``_is_excluded``
    decides that through ``parts`` and did so before.

    On POSIX ``pin_directory`` is ``O_RDONLY | O_DIRECTORY | O_NOFOLLOW``. Its
    refusal of a symlinked directory is real there and is what ``rglob`` already
    did, so the set of exported files is unchanged; the anti-rename property is
    NOT real there -- POSIX has no such lock -- and nothing here relies on it.
    Containment on POSIX rests where it always did, on ``_open_verified``
    checking the descriptor's real path.
    """
    yield from _walk_pinned(root_real, PurePath(), rel_dir.parts, keep, fenced_ok=fenced_ok)


def _walk_pinned(
    root_real: str,
    rel_dir: PurePath,
    descend: tuple[str, ...],
    keep: Callable[[PurePath], bool],
    *,
    fenced_ok: bool = False,
) -> Iterator[tuple[PurePath, int]]:
    """Pin ``root_real / rel_dir``, then walk it -- or step into *descend*'s first name.

    One frame per directory, and the frame that lists a directory is the frame
    that pinned it, so nothing is ever listed by a frame that did not verify it.
    The chain therefore starts at the crew root itself: the kernel walks that name
    to reach every candidate, so leaving it unpinned would leave the whole export
    hanging off a component that could still be renamed away.

    *descend* carries the components between the root and the tree being exported
    (``workspace``, ``plan_memory``, ``skills``). They are stepped through by NAME
    with no classification, which is safe for the same reason the walk needs no
    pre-check anywhere else: :func:`platform_compat.pin_directory` refuses a
    reparse point at the name itself, so a junction planted at ``workspace``
    fails at its own open rather than being followed.
    """
    here = os.path.join(root_real, *rel_dir.parts)
    try:
        pin = platform_compat.pin_directory(here)
    except OSError:
        return  # not a real directory, or a reparse point: refused, not followed
    try:
        if descend:
            yield from _walk_pinned(
                root_real, rel_dir / descend[0], descend[1:], keep, fenced_ok=fenced_ok
            )
            return
        try:
            with os.scandir(here) as scan:
                entries = sorted(scan, key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if _entry_is_link(entry):
                continue
            rel = rel_dir / entry.name
            if entry.is_dir(follow_symlinks=False):
                yield from _walk_pinned(root_real, rel, (), keep, fenced_ok=fenced_ok)
            elif entry.is_file(follow_symlinks=False) and keep(rel):
                fd = _open_verified(
                    os.path.join(root_real, *rel.parts), root_real, fenced_ok=fenced_ok
                )
                if fd is not None:
                    yield rel, fd
    finally:
        with contextlib.suppress(OSError):
            os.close(pin)


def _open_verified(target: str, root_real: str, *, fenced_ok: bool = False) -> int | None:
    """Open *target* without following a link at its name, and vet the descriptor.

    Split out so :func:`_open_inside` can hold the ancestor pins across the whole
    of it: the descriptor checks below are only worth anything while the path they
    were reached through is still the path that was verified.

    *fenced_ok* admits a file the agent fence (`is_sensitive_path`) covers, and is set
    for exactly one walk: ``memory_stores/``. That whole tree is fenced so a crew's agent
    cannot read another crew's memory; the export is the OPERATOR downloading their own
    install, and a private store is their memory as much as the default one is. The
    fence is not the only screen, so lifting it lifts nothing else: containment in the
    data home, the regular-file and single-link checks and the tree's own lexical filter
    all still apply, and the filter is what keeps the one credential that lives under
    the tree (the member signing key) out of the archive.
    """
    try:
        fd = platform_compat.open_file_no_reparse(target, nonblocking=True)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None
        if st.st_nlink > 1:
            # A hardlink is invisible to every path-based guard: it shares the
            # target's inode, so the fd's real path is the ALIAS's own name --
            # `fd_real_path` reports the path the descriptor was opened by, not a
            # canonical one -- and `is_sensitive_path` is then asked about an
            # innocent workspace name while the bytes behind it are a credential
            # file's. `O_NOFOLLOW` has no link to refuse, because there is no
            # symlink. Only the link COUNT, read off this descriptor, sees it.
            #
            # Refused rather than resolved: there is no way to ask "which of my
            # names is the sensitive one?", and the cost is honest and small --
            # a workspace file that legitimately has a second link is left out of
            # the export. Same rule and same reasoning as
            # `pinned_fs.refuse_hardlink_alias` and
            # `hooks.safe_read_file_bytes_nolink`.
            return None
        real = pinned_fs.fd_real_path(fd)
        if real is None:
            return None  # cannot witness containment -> fail closed
        try:
            if os.path.commonpath([real, root_real]) != root_real:
                return None
        except ValueError:  # different drives on Windows
            return None
        if not fenced_ok and is_sensitive_path(real):
            return None
    except OSError:
        return None
    else:
        held, fd = fd, -1
        return held
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)


@contextlib.contextmanager
def _open_contained_file(root_real: str, rel: PurePath) -> Iterator[int | None]:
    """Open one export file with every ancestor pinned until its descriptor closes."""
    with contextlib.ExitStack() as held:
        here = root_real
        try:
            for part in ("", *rel.parts[:-1]):
                here = os.path.join(here, part)
                pin = platform_compat.pin_directory(here)
                held.callback(os.close, pin)
        except OSError:
            yield None
            return
        fd = _open_verified(os.path.join(root_real, *rel.parts), root_real)
        if fd is not None:
            held.callback(os.close, fd)
        yield fd


# ZIP's DOS date fields have a seven-bit year and two-second resolution.
_ZIP_DATE_MIN = (1980, 1, 1, 0, 0, 0)
_ZIP_DATE_MAX = (2107, 12, 31, 23, 59, 58)


def _zip_date_time(epoch: float) -> tuple[int, int, int, int, int, int]:
    """Clamp header time only; session manifests retain the descriptor's true epoch."""
    try:
        date_time = time.localtime(epoch)[:6]
    except (OSError, OverflowError, ValueError):
        # Some platforms cannot convert negative or very large filesystem epochs.
        return _ZIP_DATE_MIN if epoch < 0 else _ZIP_DATE_MAX
    return min(_ZIP_DATE_MAX, max(_ZIP_DATE_MIN, date_time))


def _add_from_fd(zf: zipfile.ZipFile, fd: int, arcname: str) -> None:
    """Stream the bytes behind *fd* into *zf* as *arcname*.

    ``ZipFile.write`` takes a NAME and opens it itself, which is the re-open
    :func:`_open_inside` exists to remove, so the entry is built by hand instead.
    Streaming rather than reading the file whole is deliberate: a workspace file
    has no size bound here, and the export copies one of any size.

    The entry's timestamp and mode come from the same descriptor, so the metadata
    describes the bytes actually archived — not whatever the name pointed at when
    the header was built.

    ``force_zip64`` is not optional here. ``ZipFile.write`` took a name, stat'd it,
    and turned ZIP64 on by itself for a large source; a streamed entry does not know
    its size when the header is written, so without this a file over
    ``zipfile.ZIP64_LIMIT`` raises ``RuntimeError`` part-way through and the export
    endpoint answers 500. A multi-GiB file under ``workspace/`` is ordinary — a
    dataset, a model artifact — and must export, so leaving this off would trade
    one defect for another.
    """
    st = os.fstat(fd)
    info = zipfile.ZipInfo(arcname, date_time=_zip_date_time(st.st_mtime))
    info.compress_type = zf.compression
    info.external_attr = (st.st_mode & 0xFFFF) << 16
    os.lseek(fd, 0, os.SEEK_SET)
    with (
        os.fdopen(os.dup(fd), "rb", closefd=True) as src,
        zf.open(info, "w", force_zip64=True) as dest,
    ):
        shutil.copyfileobj(src, dest)


def _add_session_from_fd(
    zf: zipfile.ZipFile, fd: int, arcname: str, mtimes: dict[str, float]
) -> None:
    """Record absolute activity time from the same descriptor as the archived bytes."""
    mtime = os.fstat(fd).st_mtime
    _add_from_fd(zf, fd, arcname)
    mtimes[arcname] = mtime


#: Minimum manifest allowance in the archive's uncompressed-byte budget. The actual
#: envelope is measured too, including the three counters at their maximum width.
_MANIFEST_BASE_RESERVE = 64 * 1024
#: Per record, beyond its JSON-encoded name: a float takes at most 24 ASCII bytes;
#: indentation, colon, comma and newline take 8. This allowance covers both.
_MANIFEST_PER_MEMBER_RESERVE = 64


def _export_sessions(
    zf: zipfile.ZipFile,
    mc_real: str,
    prefix: str,
    mtimes: dict[str, float],
    *,
    manifest: dict,
) -> tuple[int, int, int]:
    """Write the persistent dashboard chats into *zf*; ``(exported, withheld, skipped)``.

    The pinned walk discovers candidate stems only. Each transcript is reopened and
    judged under its canonical history lock, which stays held through its threads
    and images so a deleted chat's private replacement cannot donate companions.

    The finished archive must pass this build's own import inventory check, so the
    chats are budgeted against the caps import enforces, `_MAX_IMPORT_MEMBERS` and
    `_MAX_IMPORT_UNCOMPRESSED`, counting every member already in *zf* and a reserve
    for the manifest still to be written. Chats are taken most recently active first
    (transcript mtime) and only whole: a chat whose files do not all fit is left out
    and counted in ``skipped``, and a later, smaller chat may still fit.
    """
    candidates: list[tuple[float, str]] = []
    for rel, fd in _walk_contained(
        mc_real, PurePath(SESSIONS_DIR_NAME), _keep_top_level_transcript
    ):
        try:
            mtime = os.fstat(fd).st_mtime
        finally:
            os.close(fd)
        stem = _chat_stem(rel.parts[1])
        if stem is not None:
            candidates.append((mtime, stem))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    # These are the only fields filled after selection; each is at most the number
    # of candidates. Measure the actual envelope (including host/user strings) with
    # those upper bounds, using the writer's indent and default ASCII escaping.
    envelope = {
        **manifest,
        "contents": {
            **manifest["contents"],
            "session_count": len(candidates),
            "sessions_withheld": len(candidates),
            "sessions_skipped_size": len(candidates),
        },
        "session_mtimes": {},
    }
    # A nonempty nested map adds a newline plus two closing-brace indent spaces.
    manifest_base = len(json.dumps(envelope, indent=2).encode("utf-8")) + 3
    manifest_bytes_left = _MAX_SETTINGS_DOCUMENT_BYTES - manifest_base
    infos = zf.infolist()
    # +1: the manifest itself is one more member, so timestamp records also stay
    # below _read_session_mtimes' _MAX_IMPORT_MEMBERS record cap.
    members_left = _MAX_IMPORT_MEMBERS - len(infos) - 1
    bytes_left = (
        _MAX_IMPORT_UNCOMPRESSED
        - sum(i.file_size for i in infos)
        - max(_MANIFEST_BASE_RESERVE, manifest_base)
    )

    history = ConversationLog(base_dir=Path(mc_real) / SESSIONS_DIR_NAME)
    exported = withheld = skipped = 0
    for _mtime, stem in candidates:
        with history.locked_stems([stem]):
            rel = PurePath(SESSIONS_DIR_NAME, f"{stem}{_TRANSCRIPT_SUFFIX}")
            with _open_contained_file(mc_real, rel) as transcript_fd:
                if transcript_fd is None:
                    continue
                if _transcript_header_verdict(_read_first_line_fd(transcript_fd)) is not None:
                    withheld += 1
                    continue
                # Sized in one pass and written in a second, both under the lock; the
                # companions open one at a time, so an image-heavy chat never holds a
                # descriptor per image.
                members = 1
                manifest_need = _manifest_entry_size(f"{prefix}/{rel.as_posix()}")
                need = os.fstat(transcript_fd).st_size
                for comp_rel, fd in _chat_companions(mc_real, stem):
                    members += 1
                    need += os.fstat(fd).st_size
                    manifest_need += _manifest_entry_size(f"{prefix}/{comp_rel.as_posix()}")
                need += manifest_need
                if (
                    members > members_left
                    or need > bytes_left
                    or manifest_need > manifest_bytes_left
                ):
                    skipped += 1
                    continue
                before = len(zf.infolist())
                _add_session_from_fd(zf, transcript_fd, f"{prefix}/{rel.as_posix()}", mtimes)
            for comp_rel, fd in _chat_companions(mc_real, stem):
                _add_session_from_fd(zf, fd, f"{prefix}/{comp_rel.as_posix()}", mtimes)
            # Charge what was actually written, not the estimate: a file that grew, or a
            # companion that appeared, between the two passes still counts against the cap.
            written = zf.infolist()[before:]
            members_left -= len(written)
            written_manifest_bytes = sum(_manifest_entry_size(i.filename) for i in written)
            manifest_bytes_left -= written_manifest_bytes
            bytes_left -= sum(i.file_size for i in written) + written_manifest_bytes
            exported += 1
    return exported, withheld, skipped


def _manifest_entry_size(arcname: str) -> int:
    """Reserve a record using the manifest writer's default JSON string escaping."""
    return len(json.dumps(arcname).encode("utf-8")) + _MANIFEST_PER_MEMBER_RESERVE


def _chat_companions(mc_real: str, stem: str) -> Iterator[tuple[PurePath, int]]:
    """``(rel, fd)`` for each companion of *stem* that travels; each fd closes on resume.

    The reply-thread sidecar, the flat image directory and the retained rotation
    segments. Callers hold the chat's history lock across the whole iteration.
    """
    sidecar = PurePath(SESSIONS_DIR_NAME, THREADS_DIR_NAME, f"{stem}{THREADS_SIDECAR_SUFFIX}")
    with _open_contained_file(mc_real, sidecar) as sidecar_fd:
        if sidecar_fd is not None:
            yield sidecar, sidecar_fd
    # sessions/<stem>.attachments/ is flat: only its direct files travel.
    images = PurePath(SESSIONS_DIR_NAME, f"{stem}{ATTACHMENTS_DIR_SUFFIX}")
    for rel, fd in _walk_contained(
        mc_real,
        images,
        lambda rel: len(rel.parts) == 3 and rel.parts[1] == f"{stem}{ATTACHMENTS_DIR_SUFFIX}",
    ):
        try:
            yield rel, fd
        finally:
            os.close(fd)
    archive = PurePath(SESSIONS_DIR_NAME, ARCHIVE_DIR_NAME)
    for rel, fd in _walk_contained(
        mc_real,
        archive,
        lambda rel: len(rel.parts) == 3 and _rotation_segment_for_stem(rel.name, stem),
    ):
        try:
            if _is_rotation_header(_read_first_line_fd(fd)):
                yield rel, fd
        finally:
            os.close(fd)


def create_export_zip(
    *,
    include_sessions: bool = False,
    memory_only: bool = False,
    include_agent_templates: bool = False,
) -> tuple[bytes, dict]:
    """Create a zip archive of Kiro Crew state. Returns (zip_bytes, manifest_dict).

    *include_sessions* adds the dashboard chats (`_export_sessions`). It is off by
    default because a transcript holds everything the agent was shown, so an archive
    that carries chats is a different thing to hand around than one that does not;
    the manifest then records ``session_count`` and ``sessions_withheld``. With it
    off the archive is unchanged.

    *memory_only* narrows the archive to the snapshot's ``memory`` component
    (`MEMORY_ONLY_FILES` and `MEMORY_ONLY_TREES`): no config, crons, hooks, skills,
    other workspace files or chats, so memory can be handed to someone else without
    the rest of the install. The manifest records ``components: ["memory"]``. It
    wins over *include_sessions*.

    *include_agent_templates* adds the agent templates the crews name, and the ones
    those name in turn (:func:`bundled_agent_templates`), under ``agents/`` with every
    MCP server ``env`` and ``headers`` value replaced by a placeholder. The manifest
    then lists them in ``agent_templates``. Off by default; with it off the archive is
    unchanged. A memory-only export carries none.
    """
    if memory_only:
        include_sessions = False
        include_agent_templates = False
    mc = _mc_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"kirocrew-export-{ts}"

    _wal_checkpoint(mc / "memory.db")
    _wal_checkpoint(mc / "memory_index.db")

    buf = io.BytesIO()
    contents_summary: dict = {}

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Core JSON/text files (none of them is memory)
        core_files: tuple[str, ...] = (
            ()
            if memory_only
            else (
                "config.json",
                *_SETTINGS_DOCUMENTS,
                "hooks.json",
                "crons.json",
                "notifications.jsonl",
                "project_dir",
                "workspace_dir",
            )
        )
        for fname in core_files:
            src = mc / fname
            if src.is_file() and not src.is_symlink():
                zf.write(str(src), f"{prefix}/{fname}")
                contents_summary[fname] = src.stat().st_size

        # The crewmate team list: one document in its own directory. Written by name like
        # the core files above -- the directory holds nothing else that rides (its lock
        # file is this host's runtime state), so the pinned tree walk below is not needed.
        teams_src = mc / "crew-teams" / "teams.json"
        if not memory_only and teams_src.is_file() and not teams_src.is_symlink():
            zf.write(str(teams_src), f"{prefix}/crew-teams/teams.json")
            contents_summary["crew-teams/teams.json"] = teams_src.stat().st_size

        # SQLite databases via backup API
        for db_name in ("memory.db", "memory_index.db"):
            src = mc / db_name
            if src.is_file() and not src.is_symlink():
                db_buf = io.BytesIO()
                _backup_sqlite(src, db_buf)
                zf.writestr(f"{prefix}/{db_name}", db_buf.getvalue())
                contents_summary[db_name] = db_buf.tell()

        # Directory trees: workspace, plan_memory, skills
        #
        # ``mc`` is resolved once, before the walk, and the walk is anchored on the
        # RESOLVED root: ``$KIROCREW_HOME`` may itself legitimately be a link, and
        # ``pin_directory`` refuses a reparse point at a name, so pinning the
        # configured spelling would return an empty archive on such a host.
        mc_real = os.path.realpath(mc)
        dir_counts: dict[str, int] = {}
        # A memory-only export walks only the memory half of the workspace; its
        # files still count as ``workspace_files`` so the import preview reads them.
        trees = MEMORY_ONLY_TREES if memory_only else ("workspace", "plan_memory", "skills")
        for dirname in trees:
            count = 0
            for rel, fd in _walk_contained(mc_real, PurePath(dirname), _keep_for_export):
                try:
                    # ``as_posix()`` for the same reason the filter rebuilds from
                    # parts: a zip member name is POSIX-separated by spec.
                    # ``ZipInfo`` happens to rewrite ``os.sep`` today, so this is
                    # not a fix for a live bug -- it stops the archive name
                    # depending on that, next to a filter that must not.
                    _add_from_fd(zf, fd, f"{prefix}/{rel.as_posix()}")
                finally:
                    os.close(fd)
                count += 1
            top = dirname.split("/", 1)[0]
            dir_counts[top] = dir_counts.get(top, 0) + count
        contents_summary["workspace_files"] = dir_counts.get("workspace", 0)
        if not memory_only:
            contents_summary["plan_memory_files"] = dir_counts.get("plan_memory", 0)
            contents_summary["skill_count"] = dir_counts.get("skills", 0)

        # Named memory stores: the same pinned walk, with the tree's own filter and the
        # fence lifted (see `_open_verified`). A store's databases go through the backup
        # API like the root ones -- the gateway holds every store open under WAL while
        # this runs -- and the filter has already dropped their sidecars.
        stores: set[str] = set()
        with hold_stores_for_read(Path(mc_real) / MEMORY_STORES_DIR_NAME):
            for rel, fd in _walk_contained(
                mc_real, PurePath(MEMORY_STORES_DIR_NAME), _keep_store_for_export, fenced_ok=True
            ):
                try:
                    arcname = f"{prefix}/{rel.as_posix()}"
                    if is_product_tree_database(rel.as_posix()):
                        real = pinned_fs.fd_real_path(fd)
                        if real is None:
                            continue
                        db_buf = io.BytesIO()
                        _backup_sqlite(Path(real), db_buf)
                        zf.writestr(arcname, db_buf.getvalue())
                    else:
                        _add_from_fd(zf, fd, arcname)
                finally:
                    os.close(fd)
                if len(rel.parts) > 1:
                    stores.add(rel.parts[1])
        contents_summary["memory_store_count"] = len(stores)

        bundled_templates: list[str] = []
        if include_agent_templates:
            carried, past_cap = bundled_agent_templates()
            for stem, spec in carried:
                zf.writestr(
                    f"{prefix}/{AGENT_TEMPLATES_DIR}/{stem}.json", json.dumps(spec, indent=2)
                )
                bundled_templates.append(stem)
            contents_summary["agent_template_count"] = len(bundled_templates)
            if past_cap:
                contents_summary["agent_templates_past_cap"] = past_cap

        # Build the envelope before selecting chats so their timestamp map is
        # budgeted against the same document the importer will read.
        manifest = {
            "version": MEMORY_EXPORT_MANIFEST_VERSION if memory_only else EXPORT_MANIFEST_VERSION,
            "format": "zip",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER", "unknown"),
            "contents": contents_summary,
        }
        if memory_only:
            manifest["components"] = ["memory"]
        if include_agent_templates:
            manifest["agent_templates"] = bundled_templates
        if include_sessions:
            session_mtimes: dict[str, float] = {}
            exported, withheld, skipped = _export_sessions(
                zf, mc_real, prefix, session_mtimes, manifest=manifest
            )
            contents_summary["session_count"] = exported
            contents_summary["sessions_withheld"] = withheld
            contents_summary["sessions_skipped_size"] = skipped
            manifest["session_mtimes"] = session_mtimes
        zf.writestr(f"{prefix}/MANIFEST.json", json.dumps(manifest, indent=2))

    return buf.getvalue(), manifest
