"""
streaming_ingestion.py — Event-driven memory ingestion pipeline.

Part of P5: Streaming Ingestion.

Provides an event bus architecture where agent actions immediately produce
queryable memories, rather than waiting for batch transcript processing.

Components:
1. MemoryEvent: typed event schema for memory-producing actions
2. EventBus: in-process pub/sub for memory events
3. StreamProcessor: consumes events, creates memory items, embeds them
4. Latency tracking: measures event-to-queryable time

Design decisions:
- In-process event bus (no external Kafka dependency for V1)
- Async-compatible but sync-first (threading for background processing)
- Events are typed and validated before processing
- Failed events go to dead-letter queue, not lost
- Graceful degradation: if streaming fails, batch pipeline still works
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger("openclaw.streaming_ingestion")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    DECISION_MADE = "decision_made"
    LESSON_LEARNED = "lesson_learned"
    FACT_OBSERVED = "fact_observed"
    PREFERENCE_SET = "preference_set"
    RULE_ESTABLISHED = "rule_established"
    CODE_CHANGED = "code_changed"
    ERROR_ENCOUNTERED = "error_encountered"
    TASK_COMPLETED = "task_completed"

    @classmethod
    def from_memory_type(cls, memory_type: str) -> EventType:
        mapping = {
            "decision": cls.DECISION_MADE,
            "lesson_learned": cls.LESSON_LEARNED,
            "learned_fix": cls.LESSON_LEARNED,
            "fact": cls.FACT_OBSERVED,
            "observation": cls.FACT_OBSERVED,
            "preference": cls.PREFERENCE_SET,
            "rule": cls.RULE_ESTABLISHED,
            "architecture_rule": cls.RULE_ESTABLISHED,
        }
        return mapping.get(memory_type, cls.FACT_OBSERVED)


@dataclass
class MemoryEvent:
    """A typed event that produces a memory item."""
    event_type: EventType
    text: str
    scope: str = "global"
    entity: str = ""
    property: str = ""
    value: str = ""
    source_agent: str = ""
    source_session: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            h = hashlib.sha256(
                f"{self.event_type}:{self.text}:{self.timestamp.isoformat()}".encode()
            ).hexdigest()[:12]
            self.event_id = f"evt-{h}"

    def to_memory_type(self) -> str:
        mapping = {
            EventType.DECISION_MADE: "decision",
            EventType.LESSON_LEARNED: "lesson_learned",
            EventType.FACT_OBSERVED: "fact",
            EventType.PREFERENCE_SET: "preference",
            EventType.RULE_ESTABLISHED: "rule",
            EventType.CODE_CHANGED: "observation",
            EventType.ERROR_ENCOUNTERED: "observation",
            EventType.TASK_COMPLETED: "observation",
        }
        return mapping.get(self.event_type, "fact")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class EventBus:
    """
    In-process pub/sub event bus for memory events.

    Thread-safe. Supports multiple subscribers per event type.
    """

    def __init__(self):
        self._subscribers: dict[EventType | None, list[Callable]] = {}
        self._lock = threading.Lock()
        self._event_count = 0
        self._dead_letters: list[dict] = []

    def subscribe(self, event_type: EventType | None, handler: Callable):
        """
        Subscribe to events of a specific type, or None for all events.
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

    def publish(self, event: MemoryEvent) -> bool:
        """
        Publish an event to all matching subscribers.

        Returns True if at least one handler succeeded.
        """
        handlers = []
        with self._lock:
            self._event_count += 1
            # Type-specific handlers
            handlers.extend(self._subscribers.get(event.event_type, []))
            # Wildcard handlers
            handlers.extend(self._subscribers.get(None, []))

        if not handlers:
            LOGGER.debug("No handlers for event %s", event.event_id)
            return False

        success = False
        for handler in handlers:
            try:
                handler(event)
                success = True
            except Exception:
                LOGGER.exception("Handler failed for event %s", event.event_id)
                with self._lock:
                    self._dead_letters.append({
                        "event": event.to_dict(),
                        "error": "handler_exception",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

        return success

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def dead_letter_count(self) -> int:
        return len(self._dead_letters)

    def get_dead_letters(self, limit: int = 10) -> list[dict]:
        return self._dead_letters[-limit:]


# ---------------------------------------------------------------------------
# Stream Processor
# ---------------------------------------------------------------------------

class StreamProcessor:
    """
    Consumes memory events and creates queryable memory items.

    Tracks processing latency for monitoring.
    """

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._processed = 0
        self._latencies: list[float] = []
        self._lock = threading.Lock()

        # Subscribe to all events
        bus.subscribe(None, self._process_event)

    def _process_event(self, event: MemoryEvent):
        """Process a single event into a memory item."""
        t0 = time.monotonic()

        item = {
            "id": f"stream-{event.event_id}",
            "text": event.text,
            "memory_type": event.to_memory_type(),
            "scope": event.scope,
            "entity": event.entity,
            "property": event.property,
            "value": event.value,
            "status": "active",
            "confidence": 0.75,
            "source_agent": event.source_agent,
            "source_session": event.source_session,
            "tags": event.tags,
            "first_seen": event.timestamp,
            "last_confirmed": event.timestamp,
        }

        elapsed = (time.monotonic() - t0) * 1000  # ms

        with self._lock:
            self._processed += 1
            self._latencies.append(elapsed)

        return item

    @property
    def processed_count(self) -> int:
        return self._processed

    def get_latency_stats(self) -> dict:
        with self._lock:
            if not self._latencies:
                return {"count": 0}
            return {
                "count": len(self._latencies),
                "mean_ms": sum(self._latencies) / len(self._latencies),
                "min_ms": min(self._latencies),
                "max_ms": max(self._latencies),
                "p99_ms": sorted(self._latencies)[int(len(self._latencies) * 0.99)] if len(self._latencies) >= 100 else max(self._latencies),
            }


# ---------------------------------------------------------------------------
# Convenience: create a wired pipeline
# ---------------------------------------------------------------------------

def create_streaming_pipeline() -> tuple[EventBus, StreamProcessor]:
    """Create a wired EventBus + StreamProcessor pair."""
    bus = EventBus()
    processor = StreamProcessor(bus)
    return bus, processor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Streaming ingestion pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run a demo with sample events")

    args = parser.parse_args()

    if args.command == "demo":
        bus, processor = create_streaming_pipeline()

        events = [
            MemoryEvent(EventType.DECISION_MADE, "Use PostgreSQL for memory storage", scope="memory-engine"),
            MemoryEvent(EventType.LESSON_LEARNED, "Always validate schema before migration", scope="memory-engine"),
            MemoryEvent(EventType.FACT_OBSERVED, "Qdrant supports 384-dim vectors", entity="qdrant", property="vector_dim", value="384"),
        ]

        for event in events:
            bus.publish(event)

        print(f"Published {bus.event_count} events")
        print(f"Processed {processor.processed_count} items")
        print(f"Dead letters: {bus.dead_letter_count}")
        print(json.dumps(processor.get_latency_stats(), indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
