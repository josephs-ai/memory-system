"""
Role-based views of memory — filters and formats memory items
based on the requesting agent's role (coder, tester, reviewer, etc.).
"""
import json
from dataclasses import dataclass
from typing import Dict, List, Any

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional import in script context
    BaseModel = object  # type: ignore[assignment]


@dataclass
class RetrievalBudget:
    max_refs: int
    max_chars: int
    preferred_scopes: List[str]
    excluded_scopes: List[str]


ROLE_BUDGETS: Dict[str, RetrievalBudget] = {
    "Planner": RetrievalBudget(
        max_refs=10,
        max_chars=20000,
        preferred_scopes=["project_state", "roadmap", "requirements"],
        excluded_scopes=["low_level_code", "test_logs"],
    ),
    "Builder": RetrievalBudget(
        max_refs=15,
        max_chars=30000,
        preferred_scopes=["current_task", "low_level_code", "api_specs", "architecture"],
        excluded_scopes=["high_level_roadmap"],
    ),
    "Reviewer": RetrievalBudget(
        max_refs=8,
        max_chars=15000,
        preferred_scopes=["pr_diffs", "coding_standards", "architecture"],
        excluded_scopes=["irrelevant_tasks"],
    ),
    "Tester": RetrievalBudget(
        max_refs=12,
        max_chars=25000,
        preferred_scopes=["test_plans", "bug_reports", "requirements", "api_specs"],
        excluded_scopes=["deployment_configs"],
    ),
    "System Designer": RetrievalBudget(
        max_refs=10,
        max_chars=25000,
        preferred_scopes=["architecture", "system_constraints", "high_level_roadmap"],
        excluded_scopes=["low_level_code", "unit_test_logs"],
    ),
    "Technical Designer": RetrievalBudget(
        max_refs=12,
        max_chars=25000,
        preferred_scopes=["api_specs", "database_schema", "architecture"],
        excluded_scopes=["high_level_roadmap", "marketing"],
    ),
}


MEMORY_PACK_FIELD_SCOPES: Dict[str, str] = {
    "project_identity": "project_state",
    "project_constraints": "requirements",
    "subproject_identity": "architecture",
    "subproject_constraints": "system_constraints",
    "current_task_context": "current_task",
    "recent_decisions": "roadmap",
    "acceptance_notes": "requirements",
    "failure_cases": "bug_reports",
    "do_not_break": "architecture",
    "open_questions": "roadmap",
}


@dataclass
class ExplainabilityTrace:
    role: str
    selected_refs: List[Dict[str, Any]]
    omitted_refs: List[Dict[str, Any]]
    scopes_used: List[str]
    budget_used_chars: int
    budget_limit_chars: int
    reasoning: List[str]

    def to_dict(self):
        return {
            "role": self.role,
            "selected_refs": self.selected_refs,
            "omitted_refs": self.omitted_refs,
            "scopes_used": self.scopes_used,
            "budget_used_chars": self.budget_used_chars,
            "budget_limit_chars": self.budget_limit_chars,
            "reasoning": self.reasoning,
        }


class MemoryRoleViewer:
    def __init__(self, role: str):
        if role not in ROLE_BUDGETS:
            raise ValueError(f"Unknown role: {role}. Available roles: {list(ROLE_BUDGETS.keys())}")
        self.role = role
        self.budget = ROLE_BUDGETS[role]

    def _coerce_contract_dict(self, retrieved_memory_contract: Any) -> Dict[str, Any]:
        if hasattr(retrieved_memory_contract, "model_dump"):
            return retrieved_memory_contract.model_dump()
        if hasattr(retrieved_memory_contract, "dict"):
            return retrieved_memory_contract.dict()
        if isinstance(retrieved_memory_contract, dict):
            return dict(retrieved_memory_contract)
        raise TypeError("retrieved_memory_contract must be a dict-like object or pydantic model")

    def _refs_from_memory_pack(self, contract_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for field_name, scope in MEMORY_PACK_FIELD_SCOPES.items():
            values = contract_dict.get(field_name) or []
            if not isinstance(values, list):
                continue
            for index, content in enumerate(values):
                text = str(content or "").strip()
                if not text:
                    continue
                refs.append(
                    {
                        "id": f"{field_name}:{index}",
                        "scope": scope,
                        "content": text,
                        "source_field": field_name,
                    }
                )
        return refs

    def _raw_refs(self, contract_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        references = contract_dict.get("references") or []
        if references:
            return list(references)
        return self._refs_from_memory_pack(contract_dict)

    def construct_view(self, retrieved_memory_contract: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs a customized prompt/system instruction combining budget rules
        with the memory_retrieval_contract.py outputs.
        Also returns the explainability trace.
        """
        contract_dict = self._coerce_contract_dict(retrieved_memory_contract)
        trace = ExplainabilityTrace(
            role=self.role,
            selected_refs=[],
            omitted_refs=[],
            scopes_used=[],
            budget_used_chars=0,
            budget_limit_chars=self.budget.max_chars,
            reasoning=[],
        )

        trace.reasoning.append(
            f"Applying budget for role {self.role}: max_refs={self.budget.max_refs}, max_chars={self.budget.max_chars}"
        )

        raw_refs = self._raw_refs(contract_dict)
        trace.reasoning.append(f"Loaded {len(raw_refs)} candidate references from retrieval contract.")

        prioritized_refs = []
        for ref in raw_refs:
            scope = ref.get("scope", "general")
            if scope in self.budget.excluded_scopes:
                trace.omitted_refs.append({"id": ref.get("id"), "reason": f"Scope {scope} is excluded for role {self.role}"})
                continue

            score = 1
            if scope in self.budget.preferred_scopes:
                score += 10
            prioritized_refs.append((score, ref))

        prioritized_refs.sort(key=lambda x: x[0], reverse=True)

        selected_content = []
        for score, ref in prioritized_refs:
            if len(trace.selected_refs) >= self.budget.max_refs:
                trace.omitted_refs.append({"id": ref.get("id"), "reason": "Max references budget reached"})
                continue

            content = ref.get("content", "")
            content_len = len(content)

            if trace.budget_used_chars + content_len > self.budget.max_chars:
                trace.omitted_refs.append({"id": ref.get("id"), "reason": "Max characters budget exceeded"})
                continue

            trace.selected_refs.append({"id": ref.get("id"), "scope": ref.get("scope")})
            if ref.get("scope") not in trace.scopes_used:
                trace.scopes_used.append(ref.get("scope"))
            trace.budget_used_chars += content_len
            selected_content.append(f"--- Scope: {ref.get('scope', 'general')} ---\n{content}\n")

        trace.reasoning.append(
            f"Selected {len(trace.selected_refs)} references using {trace.budget_used_chars}/{trace.budget_limit_chars} chars."
        )

        system_instruction = f"You are acting as the {self.role}.\n"
        system_instruction += f"Here is your customized memory context (budgeted up to {self.budget.max_chars} chars):\n\n"
        system_instruction += "\n".join(selected_content)

        return {
            "system_instruction": system_instruction,
            "explainability_trace": trace.to_dict(),
        }

    @staticmethod
    def get_explainability(trace_dict: Dict[str, Any]) -> str:
        """Feature G: A method returning a structured trace of memory selection."""
        return json.dumps(trace_dict, indent=2)
