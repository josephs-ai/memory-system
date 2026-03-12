import json
import argparse


def extract_fields(text: str):
    t = (text or "").lower()

    entity = None
    prop = None
    value = None
    memory_type_hint = None
    tags = []

    # Preferences first
    if "prefers" in t or "preference" in t:
        entity = "user_preference"
        memory_type_hint = "preference"
        tags += ["preference"]

        if "before optional polish" in t:
            prop = "build_priority"
            value = "memory_automation_before_optional_polish"

    # Checkpoint / memory system
    elif "canonical" in t and "registry" in t and ("instead of" in t or "markdown bullets" in t or "markdown bullet" in t):
        entity = "checkpoint_pipeline"
        memory_type_hint = "decision"
        prop = "canonical_memory_backend"
        value = "structured_registry_not_markdown_inference"
        tags += ["memory-system", "architecture", "registry", "markdown-inference-removed"]

    elif "cross-agent" in t or ("transcript" in t and "dehydrat" in t) or ("latest-session" in t and "transcript" in t):
        entity = "checkpoint_pipeline"
        memory_type_hint = "learned_fix"
        prop = "processing_capability"
        value = "cross_agent_supported"
        tags += ["checkpoint", "transcript", "cross-agent"]

    elif "memory" in t or "canonical" in t or "registry" in t or "checkpoint" in t or "transcript" in t or "dehydrat" in t:
        entity = "checkpoint_pipeline"
        tags += ["memory-system"]

        if "policy" in t:
            prop = "policy"
            tags += ["policy"]

    # Browser
    elif "browser" in t or "devtools" in t or "cdp" in t or "chromium" in t:
        entity = "browser"
        tags += ["browser"]

        if "tool access" in t or ("restricted" in t and "cli" in t):
            prop = "tool_access_mode"
            if "9224" in t:
                value = "cli_fallback_via_devtools_9224"
            else:
                value = "restricted_or_cli_fallback"
            tags += ["tool-access"]

        if "browser-control" in t or "browser control" in t:
            prop = prop or "capability"
            value = value or "browser_control_available"
            tags += ["capability"]
            if "cli" in t:
                tags += ["cli"]

        if "forwarded devtools port 9224" in t or ("devtools port" in t and "9224" in t):
            prop = "forwarded_devtools_port"
            value = "9224"
            tags += ["port", "devtools"]

        if "local visible chromium devtools port" in t or ("devtools port" in t and "9223" in t):
            prop = "local_devtools_port"
            value = "9223"
            tags += ["port", "devtools"]

    # Gateway
    elif "gateway" in t or "nginx" in t or "loopback" in t:
        entity = "gateway"
        tags += ["gateway"]

    # Worker
    elif "worker" in t or "browserless" in t or "vnc" in t or "upload api" in t:
        entity = "worker"
        tags += ["worker"]

    return {
        "entity": entity,
        "property": prop,
        "value": value,
        "memory_type_hint": memory_type_hint,
        "tags": sorted(set(tags)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(extract_fields(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
