"""
multimodal_memory.py — Multi-modal memory support.

Part of P6: Multi-Modal Memory.

Extends the memory system to handle non-text modalities:
- Images (screenshots, diagrams, generated art)
- Structured data (JSON schemas, configs)
- File references (with content hashes for verification)

Design:
- Text descriptions are stored in memory_items for retrieval
- Binary data stored as file references (not in DB)
- Modality-specific metadata in a dedicated schema
- Retrieval works through text descriptions (no CLIP/multimodal embeddings in V1)

V1 is description-based: images are described in text and that text is embedded.
V2 would add CLIP embeddings for direct visual similarity search.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

LOGGER = logging.getLogger("openclaw.multimodal_memory")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Modality Types
# ---------------------------------------------------------------------------

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    AUDIO = "audio"
    STRUCTURED_DATA = "structured_data"
    FILE_REFERENCE = "file_reference"


@dataclass
class MultimodalItem:
    """
    A memory item with modality-specific metadata.

    The text field contains a human-readable description used for retrieval.
    The modality_data dict contains modality-specific fields.
    """
    modality: Modality
    text_description: str
    source_path: str = ""
    content_hash: str = ""
    mime_type: str = ""
    dimensions: dict = field(default_factory=dict)  # width, height for images
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    scope: str = "global"
    entity: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_memory_item(self) -> dict:
        """Convert to a memory_items-compatible dict for storage."""
        item_id = f"mm-{hashlib.sha256(f'{self.modality}:{self.source_path}:{self.text_description[:50]}'.encode()).hexdigest()[:12]}"

        return {
            "id": item_id,
            "text": self._build_retrieval_text(),
            "memory_type": "observation",
            "scope": self.scope,
            "entity": self.entity,
            "property": f"modality:{self.modality.value}",
            "value": self.source_path or self.text_description[:50],
            "status": "active",
            "confidence": 0.7,
            "tags": [f"modality:{self.modality.value}"] + self.tags,
            "first_seen": self.created_at,
            "last_confirmed": self.created_at,
            "notes": self._build_modality_notes(),
        }

    def _build_retrieval_text(self) -> str:
        """Build the text used for vector embedding and retrieval."""
        parts = [f"[{self.modality.value.upper()}]"]
        parts.append(self.text_description)

        if self.source_path:
            filename = Path(self.source_path).name
            parts.append(f"(file: {filename})")

        if self.dimensions:
            dim_str = "x".join(str(v) for v in self.dimensions.values())
            parts.append(f"[{dim_str}]")

        return " ".join(parts)

    def _build_modality_notes(self) -> str:
        """Build modality-specific notes for the memory item."""
        notes = {
            "modality": self.modality.value,
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "mime_type": self.mime_type,
        }
        if self.dimensions:
            notes["dimensions"] = self.dimensions
        if self.metadata:
            notes["metadata"] = self.metadata

        import json
        return json.dumps(notes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modality"] = self.modality.value
        d["created_at"] = self.created_at.isoformat()
        return d


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file for content verification."""
    h = hashlib.sha256()
    path = Path(file_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_image_memory(
    description: str,
    source_path: str = "",
    *,
    width: int = 0,
    height: int = 0,
    mime_type: str = "image/png",
    tags: list[str] | None = None,
    scope: str = "global",
    entity: str = "",
) -> MultimodalItem:
    """Create a memory item for an image."""
    dimensions = {}
    if width > 0:
        dimensions["width"] = width
    if height > 0:
        dimensions["height"] = height

    content_hash = compute_file_hash(source_path) if source_path else ""

    return MultimodalItem(
        modality=Modality.IMAGE,
        text_description=description,
        source_path=source_path,
        content_hash=content_hash,
        mime_type=mime_type,
        dimensions=dimensions,
        tags=tags or [],
        scope=scope,
        entity=entity,
    )


def create_screenshot_memory(
    description: str,
    source_path: str = "",
    *,
    context: str = "",
    tags: list[str] | None = None,
    scope: str = "global",
) -> MultimodalItem:
    """Create a memory item for a screenshot."""
    metadata = {}
    if context:
        metadata["context"] = context

    content_hash = compute_file_hash(source_path) if source_path else ""

    return MultimodalItem(
        modality=Modality.SCREENSHOT,
        text_description=description,
        source_path=source_path,
        content_hash=content_hash,
        mime_type="image/png",
        tags=tags or [],
        metadata=metadata,
        scope=scope,
    )


def create_diagram_memory(
    description: str,
    source_path: str = "",
    *,
    diagram_type: str = "",
    tags: list[str] | None = None,
    scope: str = "global",
    entity: str = "",
) -> MultimodalItem:
    """Create a memory item for a diagram/architecture drawing."""
    metadata = {}
    if diagram_type:
        metadata["diagram_type"] = diagram_type

    content_hash = compute_file_hash(source_path) if source_path else ""

    return MultimodalItem(
        modality=Modality.DIAGRAM,
        text_description=description,
        source_path=source_path,
        content_hash=content_hash,
        tags=tags or [],
        metadata=metadata,
        scope=scope,
        entity=entity,
    )


def create_structured_data_memory(
    description: str,
    data: dict | list,
    *,
    data_type: str = "json",
    tags: list[str] | None = None,
    scope: str = "global",
    entity: str = "",
) -> MultimodalItem:
    """Create a memory item for structured data (JSON configs, schemas)."""
    import json
    return MultimodalItem(
        modality=Modality.STRUCTURED_DATA,
        text_description=description,
        mime_type=f"application/{data_type}",
        tags=tags or [],
        metadata={"data_preview": json.dumps(data)[:500]},
        scope=scope,
        entity=entity,
    )
