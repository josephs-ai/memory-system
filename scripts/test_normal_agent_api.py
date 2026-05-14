from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from normal_agent_api import (
    DEFAULT_ALLOWED_LIFECYCLE_STATES,
    DEFAULT_ALLOWED_MEMORY_TYPES,
    NORMAL_AGENT_REQUEST_EXAMPLE,
    NormalAgentRequest,
    RetrievalBias,
    _reset_normal_agent_packet_cache,
    build_normal_agent_packet,
    build_normal_agent_packet_response,
)


def test_normal_agent_request_accepts_task_alias_for_canonical_query():
    req = NormalAgentRequest(task="Find implementation guidance", project_id="proj-a")

    assert req.query == "Find implementation guidance"
    assert req.agent_mode == "normal"



def test_normal_agent_request_defaults_are_stable():
    req = NormalAgentRequest(query="Find implementation guidance", project_id="proj-a")

    assert req.agent_mode == "normal"
    assert req.allowed_memory_types == DEFAULT_ALLOWED_MEMORY_TYPES
    assert req.allowed_lifecycle_states == DEFAULT_ALLOWED_LIFECYCLE_STATES
    assert req.requester.requester_role == "normal_agent"
    assert req.retrieval_bias == RetrievalBias.mixed


def test_normal_agent_request_dedupes_lists_and_preserves_retrieval_knobs():
    req = NormalAgentRequest(
        query="Find implementation guidance",
        project_id="proj-a",
        domain_tags=["enemy_ai", "enemy_ai", " stealth "],
        allowed_memory_types=["decision", "decision", "guardrail"],
        allowed_lifecycle_states=["durable", "durable", "candidate"],
        need_decisions=False,
        need_patterns=True,
        need_bug_history=False,
        need_recent_session_context=True,
        retrieval_bias="fresh_only",
    )

    assert req.domain_tags == ["enemy_ai", "stealth"]
    assert req.allowed_memory_types == ["decision", "guardrail"]
    assert req.allowed_lifecycle_states == ["durable", "candidate"]
    assert req.need_decisions is False
    assert req.need_patterns is True
    assert req.need_bug_history is False
    assert req.need_recent_session_context is True
    assert req.retrieval_bias == RetrievalBias.fresh_only



def test_normal_agent_request_rejects_invalid_agent_mode():
    with pytest.raises(ValidationError):
        NormalAgentRequest(query="q", project_id="proj-a", agent_mode="planner")


def test_normal_agent_packet_compiler_surfaces_stable_task_context_and_contract_metadata():
    req = NormalAgentRequest(
        query="How should I update stealth decay?",
        project_id="proj-a",
        subproject_id="enemy_ai",
        domain_tags=["enemy_ai", "stealth"],
        max_context_items=4,
        max_context_tokens=256,
        need_decisions=True,
        need_patterns=False,
        need_bug_history=True,
        need_recent_session_context=False,
        retrieval_bias="durable_only",
    )
    rows = [
        {
            "text": "Rule: preserve escalation.",
            "path": "memory/projects/proj-a/rules.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        }
    ]

    packet = build_normal_agent_packet(req, rows, retrieval_meta={"runtime": "packet-compiler-test"})

    assert packet.task_context["query"] == "How should I update stealth decay?"
    assert packet.task_context["project_id"] == "proj-a"
    assert packet.task_context["subproject_id"] == "enemy_ai"
    assert packet.task_context["agent_mode"] == "normal"
    assert packet.task_context["domain_tags"] == ["enemy_ai", "stealth"]
    assert packet.task_context["allowed_memory_types"] == DEFAULT_ALLOWED_MEMORY_TYPES
    assert packet.task_context["allowed_lifecycle_states"] == DEFAULT_ALLOWED_LIFECYCLE_STATES
    assert packet.task_context["max_context_items"] == 4
    assert packet.task_context["max_context_tokens"] == 256
    assert packet.task_context["retrieval_bias"] == "durable_only"
    assert packet.task_context["need_decisions"] is True
    assert packet.task_context["need_patterns"] is False
    assert packet.task_context["need_bug_history"] is True
    assert packet.task_context["need_recent_session_context"] is False
    assert packet.meta["packet_contract_version"] == "2026-04-11.normal-agent.packet.v1"
    assert packet.meta["packet_compiler"] == "normal_agent_packet_compiler"



def test_build_normal_agent_packet_filters_and_compiles_sections():
    req = NormalAgentRequest(
        query="How do I preserve escalation while changing stealth decay?",
        project_id="proj-a",
        max_context_items=5,
        max_context_tokens=160,
        need_recent_session_context=True,
        allowed_memory_types=DEFAULT_ALLOWED_MEMORY_TYPES + ["open_question"],
    )
    rows = [
        {
            "text": "Do not break alert escalation outside the bounded stealth slice.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "score": 0.9,
        },
        {
            "text": "Decision: preserve the existing escalation threshold while tuning suspicion decay.",
            "path": "memory/projects/proj-a/milestone-1.md",
            "source_type": "db",
            "memory_type": "decision",
            "lifecycle_state": "reinforced",
            "score": 0.8,
        },
        {
            "text": "Implementation pattern: decay suspicion with a bounded timer rather than resetting state instantly.",
            "path": "memory/projects/proj-a/patterns.md",
            "source_type": "db",
            "memory_type": "implementation_pattern",
            "lifecycle_state": "candidate",
            "score": 0.7,
        },
        {
            "text": "Bug history: instant suspicion reset caused stealth flicker near reacquire edges.",
            "path": "memory/projects/proj-a/bugs.md",
            "source_type": "db",
            "memory_type": "bug_history",
            "lifecycle_state": "durable",
            "score": 0.6,
        },
        {
            "text": "Open question: should the decay pause during investigation?",
            "path": "memory/projects/proj-a/questions.md",
            "source_type": "db",
            "memory_type": "open_question",
            "lifecycle_state": "durable",
            "score": 0.5,
        },
    ]

    packet = build_normal_agent_packet(req, rows, retrieval_meta={"runtime": "search_runtime"})

    assert packet.status in {"fulfilled", "partial"}
    assert packet.must_follow_rules
    assert packet.relevant_decisions
    assert packet.implementation_patterns
    assert packet.known_failures_or_bug_history
    assert packet.open_questions
    assert packet.source_refs
    assert packet.meta["selected_item_count"] <= 5


def test_build_normal_agent_packet_respects_durable_only_bias():
    req = NormalAgentRequest(
        query="Need durable guidance",
        project_id="proj-a",
        retrieval_bias="durable_only",
    )
    rows = [
        {
            "text": "Candidate implementation pattern",
            "path": "memory/projects/proj-a/candidate.md",
            "source_type": "db",
            "memory_type": "implementation_pattern",
            "lifecycle_state": "candidate",
        },
        {
            "text": "Durable guardrail",
            "path": "memory/projects/proj-a/durable.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert len(packet.source_refs) == 1
    assert packet.source_refs[0].lifecycle_state == "durable"


def test_build_normal_agent_packet_prioritizes_rules_and_decisions_under_item_budget():
    req = NormalAgentRequest(
        query="Need the smallest critical packet",
        project_id="proj-a",
        max_context_items=2,
        max_context_tokens=200,
    )
    rows = [
        {
            "text": "Rule: keep the change bounded.",
            "path": "memory/projects/proj-a/rules.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "score": 0.5,
        },
        {
            "text": "Decision: preserve alert escalation.",
            "path": "memory/projects/proj-a/decisions.md",
            "source_type": "db",
            "memory_type": "decision",
            "lifecycle_state": "durable",
            "score": 0.4,
        },
        {
            "text": "Implementation pattern: use bounded timer decay.",
            "path": "memory/projects/proj-a/patterns.md",
            "source_type": "db",
            "memory_type": "implementation_pattern",
            "lifecycle_state": "durable",
            "score": 0.9,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert len(packet.source_refs) == 2
    assert packet.must_follow_rules == ["Rule: keep the change bounded."]
    assert packet.relevant_decisions == ["Decision: preserve alert escalation."]
    assert packet.implementation_patterns == []
    assert packet.meta["item_budget_exhausted"] is True



def test_build_normal_agent_packet_respects_token_budget_and_surfaces_metadata():
    req = NormalAgentRequest(
        query="Need compact critical context",
        project_id="proj-a",
        max_context_items=5,
        max_context_tokens=128,
    )
    rows = [
        {
            "text": "Rule: preserve escalation.",
            "path": "memory/projects/proj-a/rules.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "score": 0.9,
        },
        {
            "text": "Decision: keep stealth suspicion decay gradual.",
            "path": "memory/projects/proj-a/decisions.md",
            "source_type": "db",
            "memory_type": "decision",
            "lifecycle_state": "durable",
            "score": 0.8,
        },
        {
            "text": "Implementation pattern: " + "very long pattern detail " * 80,
            "path": "memory/projects/proj-a/patterns.md",
            "source_type": "db",
            "memory_type": "implementation_pattern",
            "lifecycle_state": "durable",
            "score": 1.0,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.must_follow_rules
    assert packet.relevant_decisions
    assert packet.implementation_patterns == []
    assert packet.meta["token_budget_exhausted"] is True
    assert packet.meta["skipped_for_budget_count"] >= 1



def test_build_normal_agent_packet_collapses_near_duplicates():
    req = NormalAgentRequest(
        query="Need deduped packet",
        project_id="proj-a",
        max_context_items=5,
        max_context_tokens=200,
    )
    rows = [
        {
            "text": "Rule: preserve alert escalation when tuning stealth decay.",
            "path": "memory/projects/proj-a/rules-a.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "score": 0.8,
        },
        {
            "text": "Must preserve alert escalation when tuning stealth decay.",
            "path": "memory/projects/proj-a/rules-b.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "score": 0.7,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert len(packet.must_follow_rules) == 1
    assert packet.meta["duplicates_collapsed_count"] == 1



def test_normal_agent_packet_blocks_internal_runtime_sources_by_default():
    req = NormalAgentRequest(query="Need safe packet", project_id="proj-a")
    rows = [
        {
            "text": "Rule from memory store.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        },
        {
            "text": "Internal orchestrator script content should never leak.",
            "path": "orchestrator/dispatch_runner.py",
            "source_type": "file",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert len(packet.source_refs) == 1
    assert packet.source_refs[0].path == "memory/projects/proj-a/current.md"
    assert packet.meta["blocked_internal_source_count"] == 1
    assert "orchestrator/dispatch_runner.py" in packet.meta["blocked_internal_source_paths"]



def test_normal_agent_packet_surfaces_explicit_shaping_policy_metadata():
    req = NormalAgentRequest(
        query="Need shaping policy",
        project_id="proj-a",
        max_context_items=2,
        max_context_tokens=128,
    )
    rows = [
        {
            "text": "Rule: preserve alert escalation.",
            "path": "memory/projects/proj-a/rules.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        }
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["normal_agent_shaping_policy_version"] == "2026-04-11.normal-agent.shaping.v1"
    assert packet.meta["normal_agent_shaping_policy"]["applied"] is True
    assert packet.meta["normal_agent_shaping_policy"]["budget_fields"] == {
        "max_context_items": 2,
        "max_context_tokens": 128,
    }
    assert packet.meta["normal_agent_shaping_policy"]["section_priority"][0] == "must_follow_rules"
    assert packet.meta["normal_agent_shaping_policy"]["truncate_behavior"] == "respect_item_and_token_budgets_and_surface_skips_in_meta"



def test_normal_agent_packet_surfaces_stable_normal_mode_section_contract():
    req = NormalAgentRequest(query="Need stable sections", project_id="proj-a")
    rows = [
        {
            "text": "Rule: preserve alert escalation.",
            "path": "memory/projects/proj-a/rules.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        }
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["normal_agent_section_contract_version"] == "2026-04-11.normal-agent.sections.v1"
    assert packet.meta["normal_agent_section_contract"]["agent_mode"] == "normal"
    assert packet.meta["normal_agent_section_contract"]["stable_sections"] == [
        "must_follow_rules",
        "relevant_decisions",
        "implementation_patterns",
        "recent_history",
        "known_failures_or_bug_history",
        "open_questions",
        "recommended_next_assumptions",
        "conflict_notes",
        "source_refs",
        "missing_context",
    ]



def test_normal_agent_packet_surfaces_explicit_no_script_reading_boundary():
    req = NormalAgentRequest(query="Need safe sources", project_id="proj-a")
    rows = [
        {
            "text": "Internal compiler notes.",
            "path": ".memory-index/scripts/normal_agent_api.py",
            "source_type": "file",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        }
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["normal_agent_boundary_policy_version"] == "2026-04-11.normal-agent.boundary.v1"
    assert packet.meta["normal_agent_boundary_policy"]["no_script_reading"] is True
    assert packet.meta["normal_agent_boundary_policy"]["explicit_attachment_override"] is True
    assert packet.meta["blocked_internal_source_count"] == 1



def test_normal_agent_packet_allows_explicitly_attached_internal_source():
    req = NormalAgentRequest(query="Need attached spec", project_id="proj-a")
    rows = [
        {
            "text": "Attached implementation spec that the caller explicitly authorized.",
            "path": "orchestrator/specs/public-facing-contract.md",
            "source_type": "file",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "explicitly_attached": True,
        }
    ]

    packet = build_normal_agent_packet(req, rows)

    assert len(packet.source_refs) == 1
    assert packet.meta["blocked_internal_source_count"] == 0



def test_normal_agent_packet_surfaces_explicit_normal_mode_conflict_policy_metadata():
    req = NormalAgentRequest(query="Need conflict policy", project_id="proj-a")
    rows = [
        {
            "text": "Rule: preserve alert escalation during suspicion decay tuning.",
            "path": "memory/projects/proj-a/rules-a.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "scope": "project",
        },
        {
            "text": "Rule: do not preserve alert escalation during suspicion decay tuning.",
            "path": "memory/projects/proj-a/rules-b.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "scope": "project",
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["normal_mode_conflict_policy_version"] == "2026-04-11.normal-mode.conflict.v1"
    assert packet.meta["normal_mode_conflict_policy"]["applied"] is True
    assert packet.meta["normal_mode_conflict_policy"]["winner_rules"][0] == "project_scoped_over_global"
    assert packet.meta["normal_mode_conflict_policy"]["unresolved_behavior"] == "emit_conflict_notes_and_do_not_mix_conflicting_guidance"



def test_normal_agent_packet_prefers_project_scoped_rule_over_global_pattern_on_conflict():
    req = NormalAgentRequest(query="Need coherent decay guidance", project_id="proj-a")
    rows = [
        {
            "text": "Implementation pattern: use instant suspicion reset during reacquire.",
            "path": "memory/global/patterns.md",
            "source_type": "db",
            "memory_type": "implementation_pattern",
            "lifecycle_state": "durable",
            "scope": "global",
            "updated_at": "2026-04-01T10:00:00Z",
            "score": 0.95,
        },
        {
            "text": "Rule: do not use instant suspicion reset during reacquire.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "architecture_rule",
            "lifecycle_state": "durable",
            "scope": "project",
            "updated_at": "2026-04-02T10:00:00Z",
            "score": 0.70,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.must_follow_rules == ["Rule: do not use instant suspicion reset during reacquire."]
    assert packet.implementation_patterns == []
    assert packet.conflict_notes == []
    assert packet.meta["conflicts_suppressed_count"] == 1



def test_normal_agent_packet_prefers_latest_durable_decision_over_older_candidate_note():
    req = NormalAgentRequest(query="Need one clear answer", project_id="proj-a")
    rows = [
        {
            "text": "Decision: preserve alert escalation during suspicion decay tuning.",
            "path": "memory/projects/proj-a/decisions.md",
            "source_type": "db",
            "memory_type": "decision",
            "lifecycle_state": "durable",
            "scope": "project",
            "updated_at": "2026-04-09T12:00:00Z",
            "score": 0.60,
        },
        {
            "text": "Note: do not preserve alert escalation during suspicion decay tuning.",
            "path": "memory/projects/proj-a/notes.md",
            "source_type": "db",
            "memory_type": "feature_summary",
            "lifecycle_state": "candidate",
            "scope": "project",
            "updated_at": "2026-04-01T12:00:00Z",
            "score": 0.99,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.relevant_decisions == ["Decision: preserve alert escalation during suspicion decay tuning."]
    assert packet.recommended_next_assumptions == []
    assert packet.conflict_notes == []
    assert packet.meta["conflicts_suppressed_count"] == 1



def test_normal_agent_packet_surfaces_unresolved_conflict_notes_without_mixing_context():
    req = NormalAgentRequest(query="Need safe guidance", project_id="proj-a")
    rows = [
        {
            "text": "Rule: preserve alert escalation during suspicion decay tuning.",
            "path": "memory/projects/proj-a/rules-a.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "scope": "project",
            "updated_at": "2026-04-02T10:00:00Z",
            "score": 0.80,
        },
        {
            "text": "Rule: do not preserve alert escalation during suspicion decay tuning.",
            "path": "memory/projects/proj-a/rules-b.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "scope": "project",
            "updated_at": "2026-04-02T10:00:00Z",
            "score": 0.80,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.must_follow_rules == []
    assert packet.conflict_notes == ["Conflicting guidance memories disagree about alert escalation suspicion decay tuning."]
    assert packet.meta["unresolved_conflict_count"] == 1
    assert packet.meta["conflicts_suppressed_count"] == 0



def test_normal_agent_packet_surfaces_explicit_normal_mode_retrieval_policy_metadata():
    req = NormalAgentRequest(query="Need explicit policy", project_id="proj-a")
    rows = [
        {
            "text": "Rule: preserve escalation.",
            "path": "memory/projects/proj-a/rules.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
        }
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["normal_mode_retrieval_policy_version"] == "2026-04-11.normal-mode.retrieval.v1"
    assert packet.meta["normal_mode_retrieval_policy"]["applied"] is True
    assert packet.meta["normal_mode_retrieval_policy"]["stage"] == "pre_packet_shaping"
    assert packet.meta["normal_mode_retrieval_policy"]["order"] == [
        "subproject durable memory",
        "project durable memory",
        "recent candidate/session memory",
        "archived/project history",
        "global reusable patterns",
    ]
    assert packet.meta["normal_mode_retrieval_policy"]["weighting"]["architecture_rule"] == "high"
    assert packet.meta["normal_mode_retrieval_policy"]["weighting"]["summary"] == "low_unless_sparse"



def test_normal_agent_packet_retrieval_policy_prefers_subproject_and_project_durable_before_recent_and_global():
    req = NormalAgentRequest(
        query="Need ordered packet",
        project_id="proj-a",
        subproject_id="enemy_ai",
        need_recent_session_context=True,
        max_context_items=4,
        max_context_tokens=256,
        allowed_memory_types=DEFAULT_ALLOWED_MEMORY_TYPES + ["session_summary"],
    )
    rows = [
        {
            "text": "Implementation pattern: reusable global fallback pattern.",
            "path": "memory/global/patterns.md",
            "source_type": "db",
            "memory_type": "implementation_pattern",
            "lifecycle_state": "durable",
            "scope": "global",
            "score": 0.99,
        },
        {
            "text": "Rule: subproject durable rule should lead.",
            "path": "memory/projects/proj-a/enemy_ai/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "scope": "subproject:enemy_ai",
            "score": 0.50,
        },
        {
            "text": "Decision: project durable decision comes next.",
            "path": "memory/projects/proj-a/decisions.md",
            "source_type": "db",
            "memory_type": "decision",
            "lifecycle_state": "durable",
            "scope": "project",
            "score": 0.60,
        },
        {
            "text": "Recent session context: latest candidate note.",
            "path": "memory/projects/proj-a/session.md",
            "source_type": "db",
            "memory_type": "session_summary",
            "lifecycle_state": "candidate",
            "scope": "session",
            "score": 0.95,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["retrieval_policy_order"][:4] == [
        "tier=0:subproject:durable:guardrail",
        "tier=1:project:durable:decision",
        "tier=2:session:candidate:session_summary",
        "tier=4:global:durable:implementation_pattern",
    ]
    assert packet.must_follow_rules == ["Rule: subproject durable rule should lead."]
    assert packet.relevant_decisions == ["Decision: project durable decision comes next."]



def test_normal_agent_packet_retrieval_policy_weights_rules_and_decisions_over_summaries():
    req = NormalAgentRequest(
        query="Need weighted ordering",
        project_id="proj-a",
        max_context_items=3,
        max_context_tokens=256,
        allowed_memory_types=DEFAULT_ALLOWED_MEMORY_TYPES + ["session_summary"],
    )
    rows = [
        {
            "text": "Project summary: broad summary should stay lower.",
            "path": "memory/projects/proj-a/summary.md",
            "source_type": "db",
            "memory_type": "feature_summary",
            "lifecycle_state": "durable",
            "scope": "project",
            "score": 0.99,
        },
        {
            "text": "Workflow rule: keep the patch bounded.",
            "path": "memory/projects/proj-a/workflow.md",
            "source_type": "db",
            "memory_type": "workflow_rule",
            "lifecycle_state": "durable",
            "scope": "project",
            "score": 0.60,
        },
        {
            "text": "Decision: preserve escalation behavior.",
            "path": "memory/projects/proj-a/decisions.md",
            "source_type": "db",
            "memory_type": "decision",
            "lifecycle_state": "durable",
            "scope": "project",
            "score": 0.55,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["retrieval_policy_order"][:3] == [
        "tier=1:project:durable:decision",
        "tier=1:project:durable:workflow_rule",
        "tier=1:project:durable:feature_summary",
    ]



def test_normal_agent_packet_retrieval_policy_surfaces_archive_before_global_when_needed():
    req = NormalAgentRequest(query="Need fallback ordering", project_id="proj-a")
    rows = [
        {
            "text": "Implementation pattern: global reusable fallback.",
            "path": "memory/global/patterns.md",
            "source_type": "db",
            "memory_type": "implementation_pattern",
            "lifecycle_state": "durable",
            "scope": "global",
            "score": 0.90,
        },
        {
            "text": "Bug history: archived project regression remains relevant.",
            "path": "memory/projects/proj-a/history/archive.md",
            "source_type": "db",
            "memory_type": "bug_history",
            "lifecycle_state": "durable",
            "scope": "archive",
            "score": 0.50,
        },
    ]

    packet = build_normal_agent_packet(req, rows)

    assert packet.meta["retrieval_policy_order"][:2] == [
        "tier=3:archive_or_history:durable:bug_history",
        "tier=4:global:durable:implementation_pattern",
    ]



def test_normal_agent_packet_response_surfaces_explicit_freshness_policy_metadata():
    _reset_normal_agent_packet_cache()
    req = NormalAgentRequest(query="Need explicit freshness policy", project_id="proj-a")
    rows = [
        {
            "text": "Rule: preserve alert escalation.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "updated_at": "2026-04-10T11:00:00Z",
            "score": 0.9,
        }
    ]

    response = build_normal_agent_packet_response(req, rows)

    reuse = response.packet.meta["packet_reuse"]
    assert reuse["policy_version"] == "2026-04-11.normal-mode.freshness.v1"
    assert reuse["policy"]["ttl_by_bias"]["fresh_only"] == 300
    assert "same_rows_fingerprint" in reuse["policy"]["reuse_requires"]
    assert response.meta["packet_reuse"]["policy_version"] == "2026-04-11.normal-mode.freshness.v1"



def test_normal_agent_packet_response_reuses_cached_packet_when_request_and_memory_match():
    _reset_normal_agent_packet_cache()
    req = NormalAgentRequest(query="Need cached packet", project_id="proj-a")
    rows = [
        {
            "text": "Rule: preserve alert escalation.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "updated_at": "2026-04-10T11:00:00Z",
            "score": 0.9,
        }
    ]

    first = build_normal_agent_packet_response(req, rows, retrieval_meta={"runtime": "search_runtime"})
    second = build_normal_agent_packet_response(req, rows, retrieval_meta={"runtime": "search_runtime"})

    assert first.packet.meta["packet_reuse"]["decision"] == "regenerated"
    assert first.packet.meta["packet_reuse"]["reason"] == "no_prior_cached_packet"
    assert second.packet.meta["packet_reuse"]["decision"] == "reused"
    assert second.packet.meta["packet_reuse"]["reason"] == "request_and_memory_freshness_match"
    assert second.meta["packet_reuse"]["decision"] == "reused"



def test_normal_agent_packet_response_invalidates_cache_when_memory_changes():
    _reset_normal_agent_packet_cache()
    req = NormalAgentRequest(query="Need fresh packet", project_id="proj-a")
    initial_rows = [
        {
            "text": "Rule: preserve alert escalation.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "updated_at": "2026-04-10T11:00:00Z",
            "score": 0.9,
        }
    ]
    changed_rows = [
        {
            "text": "Rule: preserve alert escalation and investigation pacing.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "durable",
            "updated_at": "2026-04-10T11:05:00Z",
            "score": 0.9,
        }
    ]

    build_normal_agent_packet_response(req, initial_rows)
    refreshed = build_normal_agent_packet_response(req, changed_rows)

    assert refreshed.packet.meta["packet_reuse"]["decision"] == "regenerated"
    assert refreshed.packet.meta["packet_reuse"]["reason"] == "memory_rows_fingerprint_changed"



def test_normal_agent_packet_response_rejects_stale_cache_entry(monkeypatch):
    _reset_normal_agent_packet_cache()
    req = NormalAgentRequest(query="Need non-stale packet", project_id="proj-a", retrieval_bias="fresh_only")
    rows = [
        {
            "text": "Rule: preserve alert escalation.",
            "path": "memory/projects/proj-a/current.md",
            "source_type": "db",
            "memory_type": "guardrail",
            "lifecycle_state": "candidate",
            "updated_at": "2026-04-10T11:00:00Z",
            "score": 0.9,
        }
    ]

    clock = iter([1000.0, 1401.0, 1401.0])
    monkeypatch.setattr("normal_agent_api._current_time", lambda: next(clock))

    first = build_normal_agent_packet_response(req, rows)
    second = build_normal_agent_packet_response(req, rows)

    assert first.packet.meta["packet_reuse"]["decision"] == "regenerated"
    assert second.packet.meta["packet_reuse"]["decision"] == "regenerated"
    assert second.packet.meta["packet_reuse"]["reason"] == "cache_ttl_expired"
    assert second.packet.meta["packet_reuse"]["ttl_seconds"] == 300



def test_normal_agent_response_builder_returns_packet():
    _reset_normal_agent_packet_cache()
    req = NormalAgentRequest(**NORMAL_AGENT_REQUEST_EXAMPLE)

    response = build_normal_agent_packet_response(
        req,
        [
            {
                "text": "Rule: keep the change bounded and preserve escalation.",
                "path": "memory/projects/mygame_prototype/current.md",
                "source_type": "db",
                "memory_type": "guardrail",
                "lifecycle_state": "durable",
                "score": 0.91,
            }
        ],
        retrieval_meta={"runtime": "search_runtime", "top_k": req.max_context_items},
    )

    assert response.ok is True
    assert response.packet.packet_id.startswith("normal-agent::mygame_prototype::")
    assert response.packet.must_follow_rules
