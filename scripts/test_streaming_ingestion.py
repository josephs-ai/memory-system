"""
Tests for streaming_ingestion.py (P5: Streaming Ingestion).
"""

import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from streaming_ingestion import (
    MemoryEvent, EventType, EventBus, StreamProcessor,
    create_streaming_pipeline,
)


class TestMemoryEvent:
    def test_auto_id(self):
        event = MemoryEvent(EventType.DECISION_MADE, "test")
        assert event.event_id.startswith("evt-")

    def test_explicit_id(self):
        event = MemoryEvent(EventType.FACT_OBSERVED, "test", event_id="custom-1")
        assert event.event_id == "custom-1"

    def test_to_memory_type(self):
        assert MemoryEvent(EventType.DECISION_MADE, "x").to_memory_type() == "decision"
        assert MemoryEvent(EventType.LESSON_LEARNED, "x").to_memory_type() == "lesson_learned"
        assert MemoryEvent(EventType.FACT_OBSERVED, "x").to_memory_type() == "fact"
        assert MemoryEvent(EventType.PREFERENCE_SET, "x").to_memory_type() == "preference"

    def test_to_dict(self):
        event = MemoryEvent(EventType.FACT_OBSERVED, "Qdrant uses 384 dims", entity="qdrant")
        d = event.to_dict()
        assert d["event_type"] == "fact_observed"
        assert d["entity"] == "qdrant"
        assert isinstance(d["timestamp"], str)

    def test_from_memory_type(self):
        assert EventType.from_memory_type("decision") == EventType.DECISION_MADE
        assert EventType.from_memory_type("unknown") == EventType.FACT_OBSERVED


class TestEventBus:
    def test_publish_and_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.FACT_OBSERVED, lambda e: received.append(e))

        event = MemoryEvent(EventType.FACT_OBSERVED, "test fact")
        bus.publish(event)

        assert len(received) == 1
        assert received[0].text == "test fact"

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        bus.publish(MemoryEvent(EventType.DECISION_MADE, "d1"))
        bus.publish(MemoryEvent(EventType.FACT_OBSERVED, "f1"))

        assert len(received) == 2

    def test_type_specific_filtering(self):
        bus = EventBus()
        decisions = []
        bus.subscribe(EventType.DECISION_MADE, lambda e: decisions.append(e))

        bus.publish(MemoryEvent(EventType.DECISION_MADE, "d1"))
        bus.publish(MemoryEvent(EventType.FACT_OBSERVED, "f1"))

        assert len(decisions) == 1

    def test_event_count(self):
        bus = EventBus()
        bus.subscribe(None, lambda e: None)
        for _ in range(5):
            bus.publish(MemoryEvent(EventType.FACT_OBSERVED, "x"))
        assert bus.event_count == 5

    def test_dead_letter_on_handler_error(self):
        bus = EventBus()
        bus.subscribe(None, lambda e: (_ for _ in ()).throw(ValueError("boom")))

        bus.publish(MemoryEvent(EventType.FACT_OBSERVED, "will fail"))
        assert bus.dead_letter_count == 1

    def test_no_handlers_returns_false(self):
        bus = EventBus()
        result = bus.publish(MemoryEvent(EventType.FACT_OBSERVED, "no one listening"))
        assert result is False

    def test_thread_safety(self):
        bus = EventBus()
        count = [0]
        lock = threading.Lock()

        def handler(e):
            with lock:
                count[0] += 1

        bus.subscribe(None, handler)

        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=lambda: bus.publish(MemoryEvent(EventType.FACT_OBSERVED, "x"))
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert count[0] == 10


class TestStreamProcessor:
    def test_processes_events(self):
        bus, processor = create_streaming_pipeline()
        bus.publish(MemoryEvent(EventType.DECISION_MADE, "Use Postgres"))
        assert processor.processed_count == 1

    def test_latency_tracking(self):
        bus, processor = create_streaming_pipeline()
        for _ in range(5):
            bus.publish(MemoryEvent(EventType.FACT_OBSERVED, "fact"))

        stats = processor.get_latency_stats()
        assert stats["count"] == 5
        assert stats["mean_ms"] >= 0
        assert stats["min_ms"] <= stats["max_ms"]

    def test_empty_latency_stats(self):
        _, processor = create_streaming_pipeline()
        assert processor.get_latency_stats() == {"count": 0}
