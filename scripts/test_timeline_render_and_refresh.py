from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from refresh_agent_memory import latest_daily_file
from render_timeline_views import load_ledger, render_daily_views



def test_render_timeline_views_writes_daily_and_latest_summary(tmp_path: Path):
    ledger = tmp_path / "events.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps({"event_id": "e1", "day_key": "2026-04-10", "timestamp": "2026-04-10T13:00:00Z", "role": "user", "text": "Pipeline Alpha must stay automated."}),
                json.dumps({"event_id": "e2", "day_key": "2026-04-10", "timestamp": "2026-04-10T13:05:00Z", "role": "assistant", "text": "Understood."}),
                json.dumps({"event_id": "e3", "day_key": "2026-04-11", "timestamp": "2026-04-11T09:00:00Z", "role": "user", "text": "Tester approved the scoped graph block."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = load_ledger(ledger)
    stats = render_daily_views(events, daily_dir=tmp_path / "daily", latest_summary_path=tmp_path / "latest.md")

    assert stats["days_rendered"] == 2
    assert stats["latest_day"] == "2026-04-11"
    assert (tmp_path / "daily" / "2026-04-10.md").exists()
    assert (tmp_path / "daily" / "2026-04-11.md").exists()
    latest_text = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert "Pipeline Alpha must stay automated." in latest_text
    assert "Tester approved the scoped graph block." in latest_text



def test_refresh_agent_memory_uses_generated_timeline_daily_dir(tmp_path: Path):
    generated_daily = tmp_path / ".memory-index" / "timeline" / "daily"
    generated_daily.mkdir(parents=True)
    older = generated_daily / "2026-04-09.md"
    newer = generated_daily / "2026-04-10.md"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")

    with patch("refresh_agent_memory.WORKSPACE", tmp_path):
        picked = latest_daily_file()

    assert picked == newer
