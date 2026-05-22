"""
Tests for multimodal_memory.py (P6: Multi-Modal Memory).
"""

import json
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multimodal_memory import (
    Modality, MultimodalItem,
    create_image_memory, create_screenshot_memory,
    create_diagram_memory, create_structured_data_memory,
    compute_file_hash,
)


class TestModality:
    def test_enum_values(self):
        assert Modality.IMAGE.value == "image"
        assert Modality.TEXT.value == "text"
        assert Modality.SCREENSHOT.value == "screenshot"


class TestMultimodalItem:
    def test_to_memory_item_schema(self):
        item = MultimodalItem(
            modality=Modality.IMAGE,
            text_description="Architecture diagram of the memory pipeline",
            source_path="/tmp/arch.png",
            scope="memory-engine",
            entity="pipeline",
        )
        mem = item.to_memory_item()

        assert mem["id"].startswith("mm-")
        assert "IMAGE" in mem["text"]
        assert "Architecture diagram" in mem["text"]
        assert mem["memory_type"] == "observation"
        assert mem["scope"] == "memory-engine"
        assert "modality:image" in mem["tags"]

    def test_retrieval_text_format(self):
        item = MultimodalItem(
            modality=Modality.SCREENSHOT,
            text_description="Bug visible in the control panel",
            source_path="/screenshots/bug.png",
            dimensions={"width": 1920, "height": 1080},
        )
        text = item._build_retrieval_text()
        assert "[SCREENSHOT]" in text
        assert "Bug visible" in text
        assert "bug.png" in text
        assert "1920x1080" in text

    def test_modality_notes_json(self):
        item = MultimodalItem(
            modality=Modality.IMAGE,
            text_description="test",
            source_path="/tmp/test.png",
            content_hash="abc123",
            mime_type="image/png",
        )
        notes = json.loads(item._build_modality_notes())
        assert notes["modality"] == "image"
        assert notes["content_hash"] == "abc123"

    def test_to_dict(self):
        item = MultimodalItem(Modality.TEXT, "simple text")
        d = item.to_dict()
        assert d["modality"] == "text"
        assert isinstance(d["created_at"], str)


class TestComputeFileHash:
    def test_real_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content for hashing")
            f.flush()
            h = compute_file_hash(f.name)
            assert len(h) == 16
            # Deterministic
            assert compute_file_hash(f.name) == h

    def test_nonexistent_file(self):
        assert compute_file_hash("/nonexistent/path.png") == ""


class TestFactoryFunctions:
    def test_create_image_memory(self):
        item = create_image_memory(
            "Ronin sprite with sword animation",
            width=128, height=128,
            tags=["game-art"],
            scope="pixel-ronin",
        )
        assert item.modality == Modality.IMAGE
        assert item.dimensions == {"width": 128, "height": 128}
        mem = item.to_memory_item()
        assert "modality:image" in mem["tags"]
        assert "game-art" in mem["tags"]

    def test_create_screenshot_memory(self):
        item = create_screenshot_memory(
            "Control panel showing memory stats",
            context="debugging retrieval quality",
        )
        assert item.modality == Modality.SCREENSHOT
        assert item.metadata["context"] == "debugging retrieval quality"

    def test_create_diagram_memory(self):
        item = create_diagram_memory(
            "Memory pipeline data flow diagram",
            diagram_type="flowchart",
            entity="pipeline",
        )
        assert item.modality == Modality.DIAGRAM
        assert item.metadata["diagram_type"] == "flowchart"

    def test_create_structured_data_memory(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        item = create_structured_data_memory(
            "User profile JSON schema",
            schema,
            data_type="json-schema",
        )
        assert item.modality == Modality.STRUCTURED_DATA
        assert "data_preview" in item.metadata
        mem = item.to_memory_item()
        assert "modality:structured_data" in mem["tags"]
