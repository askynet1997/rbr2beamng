from __future__ import annotations

import contextvars
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .filesystem import FileSandbox, SandboxViolationError, current_filesystem

PROFILE_SCHEMA_VERSION = 1
PROFILE_HISTORY = 4
_ACTIVE_PROFILE: contextvars.ContextVar[ConversionProfiler | None] = (
    contextvars.ContextVar("rbr2beamng_profile", default=None)
)


def _json_value(value: Any) -> Any:
    if isinstance(value, FileSandbox):
        return {
            "roots": [
                {
                    "path": str(root.path),
                    "writable": root.writable,
                }
                for root in value.roots
            ]
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return result[:80] or "stage"


def _profile_timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        timestamp = value.astimezone(timezone.utc)
    else:
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return _slug(value)
    return timestamp.strftime("%Y%m%dT%H%M%S.%fZ")


def _profile_filename(
    value: str,
    timestamp: datetime | str,
    suffix: str = "",
) -> str:
    stem = _slug(value)
    if not stem.startswith("rbr_"):
        stem = f"rbr_{stem}"
    return f"{stem}_{_profile_timestamp(timestamp)}{suffix}.json"


def profile_directory(config_directory: Path) -> Path:
    override = os.environ.get("RBR2BEAMNG_PROFILE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return config_directory.expanduser().resolve() / "profiles"


def _write_json_atomic(destination: Path, payload: Any) -> bool:
    filesystem = current_filesystem()
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        filesystem.mkdir(destination.parent, parents=True, exist_ok=True)
        filesystem.write_text(
            temporary,
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        filesystem.replace(temporary, destination)
        return True
    except SandboxViolationError:
        raise
    except Exception:
        try:
            filesystem.unlink(temporary, missing_ok=True)
        except OSError:
            pass
        return False


def _open_lock_file(path: Path, *, blocking: bool) -> BinaryIO | None:
    filesystem = current_filesystem()
    stream: BinaryIO | None = None
    try:
        filesystem.mkdir(path.parent, parents=True, exist_ok=True)
        stream = filesystem.open(path, "a+b")
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(
                stream.fileno(),
                msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK,
                1,
            )
        else:
            import fcntl

            operation = fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            fcntl.flock(stream.fileno(), operation)
        return stream
    except (ImportError, OSError):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        return None


def _close_lock_file(stream: BinaryIO) -> None:
    try:
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


@contextmanager
def _rotation_lock(directory: Path) -> Iterator[None]:
    stream = _open_lock_file(directory / ".rotation.lock", blocking=True)
    try:
        yield
    finally:
        if stream is not None:
            _close_lock_file(stream)


def _posix_memory() -> dict[str, int] | None:
    if sys.platform == "win32":
        return None
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak *= 1024
        current = 0
        status = Path("/proc/self/statm")
        filesystem = current_filesystem()
        if filesystem.is_file(status):
            pages = int(
                filesystem.read_text(status, encoding="ascii").split()[1]
            )
            current = pages * int(os.sysconf("SC_PAGE_SIZE"))
        return {
            "workingSetBytes": current,
            "privateBytes": 0,
            "pagefileBytes": 0,
            "peakWorkingSetBytes": peak,
            "peakCommitBytes": 0,
        }
    except (OSError, ValueError):
        return None


def process_memory() -> dict[str, int]:
    fallback = {
        "workingSetBytes": 0,
        "privateBytes": 0,
        "pagefileBytes": 0,
        "peakWorkingSetBytes": 0,
        "peakCommitBytes": 0,
    }
    try:
        return _posix_memory() or fallback
    except Exception:
        return fallback


@dataclass
class ProfileSpan:
    span_id: int
    name: str
    category: str
    parent_id: int | None
    data: dict[str, Any]
    started_at: float = field(default_factory=time.perf_counter)
    cpu_started_at: float = field(default_factory=time.process_time)
    memory_started: dict[str, int] = field(default_factory=process_memory)

    def update(self, **values: Any) -> None:
        self.data.update(values)


class _NoOpProfileSpan:
    __slots__ = ()

    def update(self, **_values: Any) -> None:
        pass


_NO_OP_PROFILE_SPAN = _NoOpProfileSpan()


class ConversionProfiler:
    def __init__(
        self,
        directory: Path,
        *,
        stage: str,
        source_format: str,
        converter_version: str,
        options: Any,
    ) -> None:
        self.directory = directory
        self.stage = stage
        self.source_format = source_format
        self.converter_version = converter_version
        self.options = _json_value(options)
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"_{os.getpid()}_{_slug(stage)}"
        )
        self.started_utc = datetime.now(timezone.utc)
        self.started_at = time.perf_counter()
        self.cpu_started_at = time.process_time()
        self.memory_started = process_memory()
        self.spans: list[dict[str, Any]] = []
        self.progress: list[dict[str, Any]] = []
        self._span_stack: list[int] = []
        self._next_span_id = 1
        self._last_progress_at = self.started_at
        self.result: dict[str, Any] = {}
        self.output_path: Path | None = None
        self.warning_count = 0
        self.profile_path: Path | None = None
        self._finished = False
        self._last_marker_at = 0.0
        self.marker_path = self.directory / f"running_{self.run_id}.json"
        self.marker_lock_path = self.marker_path.with_suffix(".lock")
        self._marker_lock = _open_lock_file(
            self.marker_lock_path,
            blocking=True,
        )
        self._recover_interrupted()
        self._write_marker(force=True)

    def _write_marker(
        self,
        *,
        force: bool = False,
        last_progress: dict[str, Any] | None = None,
    ) -> None:
        now = time.perf_counter()
        if not force and now - self._last_marker_at < 1.0:
            return
        if self._marker_lock is None:
            return
        if _write_json_atomic(
            self.marker_path,
            {
                "schemaVersion": PROFILE_SCHEMA_VERSION,
                "runId": self.run_id,
                "pid": os.getpid(),
                "stage": self.stage,
                "sourceFormat": self.source_format,
                "converterVersion": self.converter_version,
                "startedUtc": self.started_utc.isoformat(),
                "elapsedSeconds": round(now - self.started_at, 6),
                "lastProgress": last_progress,
                "memory": process_memory(),
            },
        ):
            self._last_marker_at = now

    def _recover_interrupted(self) -> None:
        filesystem = current_filesystem()
        try:
            filesystem.mkdir(self.directory, parents=True, exist_ok=True)
            markers = tuple(filesystem.glob(self.directory, "running_*.json"))
        except OSError:
            return
        recovered = False
        for marker in markers:
            lock_path = marker.with_suffix(".lock")
            try:
                if not filesystem.is_file(lock_path):
                    continue
            except OSError:
                continue
            lock = _open_lock_file(lock_path, blocking=False)
            if lock is None:
                continue
            remove_lock = False
            try:
                value = json.loads(
                    filesystem.read_text(marker, encoding="utf-8")
                )
                run_id = str(value["runId"])
                pid = int(value["pid"])
                destination = self.directory / _profile_filename(
                    str(value.get("stage") or "stage"),
                    str(value.get("startedUtc") or run_id),
                    "_interrupted",
                )
                if filesystem.exists(destination):
                    filesystem.unlink(marker, missing_ok=True)
                    remove_lock = True
                    continue
                ended_utc = datetime.now(timezone.utc)
                elapsed = float(value.get("elapsedSeconds") or 0.0)
                payload = {
                    "schemaVersion": PROFILE_SCHEMA_VERSION,
                    "run": {
                        "id": run_id,
                        "pid": pid,
                        "stage": value.get("stage"),
                        "sourceFormat": value.get("sourceFormat"),
                        "converterVersion": value.get("converterVersion"),
                        "status": "interrupted",
                        "startedUtc": value.get("startedUtc"),
                        "endedUtc": ended_utc.isoformat(),
                        "durationSeconds": elapsed,
                        "cpuSeconds": None,
                        "options": None,
                        "outputPath": None,
                        "warningCount": 0,
                        "result": {},
                        "error": {
                            "type": "ProcessInterrupted",
                            "message": (
                                "The conversion process ended without "
                                "finalizing its profile."
                            ),
                        },
                    },
                    "memory": {
                        "start": None,
                        "end": value.get("memory"),
                    },
                    "progress": (
                        [value["lastProgress"]]
                        if value.get("lastProgress")
                        else []
                    ),
                    "spans": [],
                    "summary": {
                        "operations": {},
                        "slowestSpans": [],
                        "slowestProgressIntervals": (
                            [value["lastProgress"]]
                            if value.get("lastProgress")
                            else []
                        ),
                    },
                }
                if not _write_json_atomic(destination, payload):
                    continue
                filesystem.unlink(marker, missing_ok=True)
                remove_lock = True
                recovered = True
            except Exception:
                continue
            finally:
                _close_lock_file(lock)
                if remove_lock:
                    try:
                        filesystem.unlink(lock_path, missing_ok=True)
                    except OSError:
                        pass
        if recovered:
            self._rotate()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str = "conversion",
        **data: Any,
    ) -> Iterator[ProfileSpan]:
        span = ProfileSpan(
            span_id=self._next_span_id,
            name=name,
            category=category,
            parent_id=self._span_stack[-1] if self._span_stack else None,
            data=dict(data),
        )
        self._next_span_id += 1
        self._span_stack.append(span.span_id)
        status = "ok"
        error: dict[str, str] | None = None
        try:
            yield span
        except BaseException as exc:
            status = "error"
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            if self._span_stack and self._span_stack[-1] == span.span_id:
                self._span_stack.pop()
            try:
                ended_at = time.perf_counter()
                memory_ended = process_memory()
                record: dict[str, Any] = {
                    "id": span.span_id,
                    "parentId": span.parent_id,
                    "category": span.category,
                    "name": span.name,
                    "status": status,
                    "startSeconds": round(
                        span.started_at - self.started_at,
                        6,
                    ),
                    "durationSeconds": round(
                        ended_at - span.started_at,
                        6,
                    ),
                    "cpuSeconds": round(
                        time.process_time() - span.cpu_started_at,
                        6,
                    ),
                    "memoryStart": span.memory_started,
                    "memoryEnd": memory_ended,
                    "data": _json_value(span.data),
                }
                if error:
                    record["error"] = error
                self.spans.append(record)
            except Exception:
                pass

    def record_progress(
        self,
        phase: str,
        message: str,
        **data: Any,
    ) -> None:
        now = time.perf_counter()
        event = {
            "sequence": len(self.progress) + 1,
            "phase": phase,
            "message": message,
            "elapsedSeconds": round(now - self.started_at, 6),
            "sincePreviousSeconds": round(
                now - self._last_progress_at,
                6,
            ),
            "data": _json_value(
                {
                    key: value
                    for key, value in data.items()
                    if value is not None
                }
            ),
        }
        self.progress.append(event)
        self._last_progress_at = now
        self._write_marker(last_progress=event)

    def set_result(
        self,
        *,
        output_path: Path | None = None,
        warning_count: int = 0,
        **values: Any,
    ) -> None:
        self.output_path = output_path
        self.warning_count = warning_count
        self.result.update(values)

    def _summary(self) -> dict[str, Any]:
        operations: dict[str, dict[str, float | int]] = {}
        child_seconds: dict[int, float] = {}
        for span in self.spans:
            parent_id = span["parentId"]
            if parent_id is not None:
                child_seconds[parent_id] = (
                    child_seconds.get(parent_id, 0.0)
                    + float(span["durationSeconds"])
                )
        for span in self.spans:
            key = f"{span['category']}.{span['name']}"
            entry = operations.setdefault(
                key,
                {
                    "count": 0,
                    "okCount": 0,
                    "errorCount": 0,
                    "totalSeconds": 0.0,
                    "totalExclusiveSeconds": 0.0,
                    "maxSeconds": 0.0,
                    "maxExclusiveSeconds": 0.0,
                },
            )
            duration = float(span["durationSeconds"])
            exclusive = max(
                0.0,
                duration - child_seconds.get(int(span["id"]), 0.0),
            )
            entry["count"] = int(entry["count"]) + 1
            status_key = (
                "okCount"
                if span["status"] == "ok"
                else "errorCount"
            )
            entry[status_key] = int(entry[status_key]) + 1
            entry["totalSeconds"] = round(
                float(entry["totalSeconds"]) + duration,
                6,
            )
            entry["totalExclusiveSeconds"] = round(
                float(entry["totalExclusiveSeconds"]) + exclusive,
                6,
            )
            entry["maxSeconds"] = max(
                float(entry["maxSeconds"]),
                duration,
            )
            entry["maxExclusiveSeconds"] = max(
                float(entry["maxExclusiveSeconds"]),
                exclusive,
            )
        for entry in operations.values():
            entry["averageSeconds"] = round(
                float(entry["totalSeconds"]) / max(1, int(entry["count"])),
                6,
            )
            entry["averageExclusiveSeconds"] = round(
                float(entry["totalExclusiveSeconds"])
                / max(1, int(entry["count"])),
                6,
            )
        slowest_spans = sorted(
            self.spans,
            key=lambda item: float(item["durationSeconds"]),
            reverse=True,
        )[:50]
        slowest_progress = sorted(
            self.progress,
            key=lambda item: float(item["sincePreviousSeconds"]),
            reverse=True,
        )[:50]
        return {
            "operations": operations,
            "slowestSpans": [
                {
                    "id": span["id"],
                    "category": span["category"],
                    "name": span["name"],
                    "status": span["status"],
                    "durationSeconds": span["durationSeconds"],
                    "exclusiveSeconds": round(
                        max(
                            0.0,
                            float(span["durationSeconds"])
                            - child_seconds.get(int(span["id"]), 0.0),
                        ),
                        6,
                    ),
                    "data": span["data"],
                    "error": span.get("error"),
                }
                for span in slowest_spans
            ],
            "slowestProgressIntervals": slowest_progress,
        }

    def finish(
        self,
        status: str,
        error: BaseException | None = None,
    ) -> Path | None:
        if self._finished:
            return self.profile_path
        self._finished = True
        try:
            return self._finish(status, error)
        except SandboxViolationError:
            raise
        except Exception:
            return None
        finally:
            try:
                current_filesystem().unlink(self.marker_path, missing_ok=True)
            except SandboxViolationError:
                raise
            except OSError:
                pass
            finally:
                if self._marker_lock is not None:
                    _close_lock_file(self._marker_lock)
                    self._marker_lock = None
                try:
                    current_filesystem().unlink(
                        self.marker_lock_path,
                        missing_ok=True,
                    )
                except SandboxViolationError:
                    raise
                except OSError:
                    pass

    def _finish(
        self,
        status: str,
        error: BaseException | None = None,
    ) -> Path | None:
        ended_at = time.perf_counter()
        ended_utc = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "schemaVersion": PROFILE_SCHEMA_VERSION,
            "run": {
                "id": self.run_id,
                "pid": os.getpid(),
                "stage": self.stage,
                "sourceFormat": self.source_format,
                "converterVersion": self.converter_version,
                "platform": sys.platform,
                "architecture": platform.machine(),
                "pythonVersion": platform.python_version(),
                "cpuCount": os.cpu_count(),
                "frozen": bool(getattr(sys, "frozen", False)),
                "status": status,
                "startedUtc": self.started_utc.isoformat(),
                "endedUtc": ended_utc.isoformat(),
                "durationSeconds": round(ended_at - self.started_at, 6),
                "cpuSeconds": round(
                    time.process_time() - self.cpu_started_at,
                    6,
                ),
                "options": self.options,
                "outputPath": (
                    str(self.output_path)
                    if self.output_path is not None
                    else None
                ),
                "warningCount": self.warning_count,
                "result": _json_value(self.result),
            },
            "memory": {
                "start": self.memory_started,
                "end": process_memory(),
            },
            "progress": self.progress,
            "spans": self.spans,
            "summary": self._summary(),
        }
        if error is not None:
            payload["run"]["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                ),
            }
        profile_name = (
            self.output_path.stem
            if self.output_path is not None
            else str(self.result.get("levelId") or self.stage)
        )
        destination = self.directory / _profile_filename(
            profile_name,
            self.started_utc,
        )
        if not _write_json_atomic(destination, payload):
            return None
        self.profile_path = destination
        self._rotate()
        return destination

    def _rotate(self) -> None:
        filesystem = current_filesystem()
        with _rotation_lock(self.directory):
            try:
                profiles = sorted(
                    filesystem.glob(self.directory, "conversion_*.json"),
                    key=lambda path: (
                        filesystem.stat(path).st_mtime_ns,
                        path.name,
                    ),
                    reverse=True,
                )
            except OSError:
                return
            for stale in profiles[PROFILE_HISTORY:]:
                try:
                    filesystem.unlink(stale)
                except SandboxViolationError:
                    raise
                except OSError:
                    pass


def active_profile() -> ConversionProfiler | None:
    return _ACTIVE_PROFILE.get()


@contextmanager
def profile_conversion(profile: ConversionProfiler) -> Iterator[ConversionProfiler]:
    token = _ACTIVE_PROFILE.set(profile)
    try:
        yield profile
    except BaseException as exc:
        profile.finish(
            "cancelled"
            if type(exc).__name__ == "ConversionCancelled"
            else "error",
            exc,
        )
        raise
    else:
        profile.finish("success")
    finally:
        _ACTIVE_PROFILE.reset(token)


@contextmanager
def profile_span(
    name: str,
    *,
    category: str = "conversion",
    **data: Any,
) -> Iterator[ProfileSpan | _NoOpProfileSpan]:
    profile = active_profile()
    if profile is None:
        yield _NO_OP_PROFILE_SPAN
        return
    with profile.span(name, category=category, **data) as span:
        yield span


def record_progress(
    phase: str,
    message: str,
    **data: Any,
) -> None:
    profile = active_profile()
    if profile is not None:
        try:
            profile.record_progress(phase, message, **data)
        except Exception:
            pass
