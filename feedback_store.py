import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency in local test environments
    psycopg = None


CREATE_FEEDBACK_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id TEXT NOT NULL,
    locale TEXT NOT NULL,
    surface TEXT NOT NULL,
    task_type TEXT NOT NULL,
    task_id TEXT NULL,
    run_id TEXT NULL,
    checkpoint_id TEXT NULL,
    prompt_type TEXT NULL,
    presented_payload JSONB NOT NULL,
    response_payload JSONB NOT NULL,
    response_label TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    app_version TEXT NOT NULL
);
"""


@dataclass
class StoredFeedbackEvent:
    session_id: str
    locale: str
    surface: str
    task_type: str
    task_id: Optional[str]
    run_id: Optional[str]
    checkpoint_id: Optional[str]
    prompt_type: Optional[str]
    presented_payload: Dict[str, Any]
    response_payload: Dict[str, Any]
    response_label: str
    latency_ms: int
    app_version: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_record(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "locale": self.locale,
            "surface": self.surface,
            "task_type": self.task_type,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "prompt_type": self.prompt_type,
            "presented_payload": dict(self.presented_payload),
            "response_payload": dict(self.response_payload),
            "response_label": self.response_label,
            "latency_ms": self.latency_ms,
            "app_version": self.app_version,
        }


class BaseFeedbackStore:
    def is_available(self) -> bool:
        raise NotImplementedError

    def detail(self) -> str:
        raise NotImplementedError

    def record_event(self, event: StoredFeedbackEvent) -> Dict[str, Any]:
        raise NotImplementedError


class UnavailableFeedbackStore(BaseFeedbackStore):
    def __init__(self, detail: str):
        self._detail = detail

    def is_available(self) -> bool:
        return False

    def detail(self) -> str:
        return self._detail

    def record_event(self, event: StoredFeedbackEvent) -> Dict[str, Any]:
        raise RuntimeError(self._detail)


class MemoryFeedbackStore(BaseFeedbackStore):
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        return True

    def detail(self) -> str:
        return "Memory feedback store is active."

    def record_event(self, event: StoredFeedbackEvent) -> Dict[str, Any]:
        payload = event.to_record()
        self.events.append(payload)
        return payload


class PostgresFeedbackStore(BaseFeedbackStore):
    def __init__(self, database_url: str, auto_init: bool = False):
        if psycopg is None:
            raise RuntimeError(
                "psycopg is not installed. Install requirements with the postgres extras to enable feedback storage."
            )
        self.database_url = database_url
        if auto_init:
            self.ensure_schema()

    def is_available(self) -> bool:
        return True

    def detail(self) -> str:
        return "Managed Postgres feedback store is active."

    def ensure_schema(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(CREATE_FEEDBACK_EVENTS_SQL)
            connection.commit()

    def record_event(self, event: StoredFeedbackEvent) -> Dict[str, Any]:
        payload = event.to_record()
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO feedback_events (
                        id,
                        session_id,
                        locale,
                        surface,
                        task_type,
                        task_id,
                        run_id,
                        checkpoint_id,
                        prompt_type,
                        presented_payload,
                        response_payload,
                        response_label,
                        latency_ms,
                        app_version
                    ) VALUES (
                        %(id)s,
                        %(session_id)s,
                        %(locale)s,
                        %(surface)s,
                        %(task_type)s,
                        %(task_id)s,
                        %(run_id)s,
                        %(checkpoint_id)s,
                        %(prompt_type)s,
                        %(presented_payload)s::jsonb,
                        %(response_payload)s::jsonb,
                        %(response_label)s,
                        %(latency_ms)s,
                        %(app_version)s
                    )
                    """,
                    {
                        **payload,
                        "presented_payload": json.dumps(payload["presented_payload"], ensure_ascii=False),
                        "response_payload": json.dumps(payload["response_payload"], ensure_ascii=False),
                    },
                )
            connection.commit()
        return payload


def build_feedback_store(database_url: Optional[str], auto_init: bool = False) -> BaseFeedbackStore:
    if not database_url:
        return UnavailableFeedbackStore("Feedback storage is not configured.")

    if psycopg is None:
        return UnavailableFeedbackStore(
            "Feedback storage requires psycopg. Install dependencies before enabling the managed store."
        )

    try:
        return PostgresFeedbackStore(database_url=database_url, auto_init=auto_init)
    except Exception as exc:  # noqa: BLE001 - keep startup diagnostics simple
        return UnavailableFeedbackStore(f"Feedback storage failed to initialize: {type(exc).__name__}: {exc}")
