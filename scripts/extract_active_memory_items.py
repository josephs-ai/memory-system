"""
Extract active memory items from source data.

Key functions: infer_fields, make_id, main
"""
import json
import argparse
import hashlib
from pathlib import Path

def infer_fields(text: str):
    t = text.lower()

    memory_type = "fact"
    scope = "stable"
    entity = "openclaw"
    prop = None
    value = None
    tags = []

    if "browser" in t or "devtools" in t or "cdp" in t or "chromium" in t:
        entity = "browser"
        tags.append("browser")

    if "gateway" in t or "nginx" in t or "loopback" in t or "18789" in t:
        entity = "gateway"
        tags.append("gateway")

    if "worker" in t or "browserless" in t or "vnc" in t or "upload api" in t or "couchdb" in t:
        entity = "worker"
        tags.append("worker")

    if "prefer" in t or "joseph prefers" in t:
        memory_type = "preference"
        entity = "user_preference"
        tags.append("preference")

    if "9224" in t:
        prop = "forwarded_devtools_port"
        value = "9224"
        tags.append("port")

    if "9223" in t:
        prop = "local_devtools_port"
        value = "9223"
        tags.append("port")

    if "tool access" in t or "cli fallback" in t or "restricted" in t:
        prop = "tool_access_mode"
        if "cli fallback" in t:
            value = "cli_fallback_via_devtools_9224"
        elif "unavailable" in t:
            value = "unavailable_in_session"
        tags.append("tool-access")

    if "browser-control capability" in t or "browser control" in t:
        prop = prop or "capability"
        value = value or "browser_control_available"
        tags.append("capability")

    return memory_type, scope, entity, prop, value, sorted(set(tags))

def make_id(target_path: str, text: str):
    h = hashlib.sha1(f"{target_path}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"existing-{h}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    path = Path(args.target)
    lines = path.read_text(encoding="utf-8").splitlines()

    current_section = None
    items = []

    for line in lines:
        s = line.strip()

        if s.startswith("## "):
            current_section = s[3:].strip()
            continue

        if not s.startswith("- "):
            continue

        if current_section == "Superseded":
            continue

        text = s[2:].strip()
        memory_type, scope, entity, prop, value, tags = infer_fields(text)

        item = {
            "id": make_id(str(path), text),
            "text": text,
            "memory_type": memory_type,
            "scope": scope,
            "entity": entity,
            "property": prop,
            "value": value,
            "status": "active",
            "confidence": 0.75,
            "source_agent": "canonical_memory",
            "source_session": str(path),
            "source_chunk": current_section or "root",
            "first_seen": None,
            "last_confirmed": None,
            "supersedes": None,
            "tags": tags,
            "notes": f"Derived from active bullet in {path.name}"
        }
        items.append(item)

    if not items:
        print("NONE")
        return

    for item in items:
        print(json.dumps(item, ensure_ascii=False))

if __name__ == "__main__":
    main()
