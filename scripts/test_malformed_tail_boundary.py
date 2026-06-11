import pytest
import json
from pathlib import Path
from dehydrate_transcript import load_events

def test_malformed_jsonl_tail_ignored(tmp_path):
    # Create a transcript with 2 good lines and 1 malformed tail
    transcript = tmp_path / "session.jsonl"
    lines = [
        json.dumps({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}}),
        json.dumps({"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}}),
        '{"type": "message", "message": {"role": "user", "content": [{"type": "text", "te' # malformed tail
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    events = load_events(transcript, offset=0)
    assert len(events) == 2
    assert events[0]["message"]["role"] == "user"
    assert events[1]["message"]["role"] == "assistant"

