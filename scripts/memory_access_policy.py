from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"

AGENT_ACCESS = {
    "coder": {"public", "internal"},
    "review": {"public", "internal"},
    "art": {"public"},
    "default": {"public"},
    "security": {"public", "internal", "restricted"},
    "admin": {"public", "internal", "restricted"},
}

EXPLICIT_AGENT_ACCESS = {
    "main": {"public", "internal", "restricted"},
}

def infer_agent_class(agent_name: str) -> str:
    name = (agent_name or "").strip().lower()

    if any(x in name for x in ["security", "secops", "audit", "admin"]):
        return "security" if "security" in name or "secops" in name else "admin"
    if any(x in name for x in ["builder", "coder", "dev", "engineer", "tester", "implementer", "programmer"]):
        return "coder"
    if any(x in name for x in ["review", "reviewer", "critic"]):
        return "review"
    if any(x in name for x in ["artist", "art", "design", "concept", "visual"]):
        return "art"
    return "default"

def allowed_sensitivities(agent_name: str):
    if agent_name in EXPLICIT_AGENT_ACCESS:
        return EXPLICIT_AGENT_ACCESS[agent_name]
    klass = infer_agent_class(agent_name)
    return AGENT_ACCESS.get(klass, {"public"})
