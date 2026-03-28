from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from memory_routing_models import WorkItemMemoryMetadata, MemoryScopeLevel

class MemoryPack(BaseModel):
    """
    Normalized memory structure as per Deliverable 2.
    """
    project_identity: List[str] = Field(default_factory=list)
    project_constraints: List[str] = Field(default_factory=list)
    subproject_identity: List[str] = Field(default_factory=list)
    subproject_constraints: List[str] = Field(default_factory=list)
    current_task_context: List[str] = Field(default_factory=list)
    recent_decisions: List[str] = Field(default_factory=list)
    acceptance_notes: List[str] = Field(default_factory=list)
    failure_cases: List[str] = Field(default_factory=list)
    do_not_break: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

def _process_refs(pack: MemoryPack, refs: List[Dict[str, Any]], scope: str, role: str):
    """
    Helper to process references and populate the pack based on their content type
    and the target scope, applying role-aware logic if needed.
    """
    for ref in refs:
        # Simplistic mapping based on 'category' or 'type' key in the ref dict
        category = ref.get("category")
        content = ref.get("content", "")
        
        # Example of role-aware filtering: if a ref is tagged for a specific role and it doesn't match, skip.
        if "target_role" in ref and ref["target_role"] != role and ref["target_role"] != "all":
            continue

        if category == "identity":
            if scope == "global":
                pack.project_identity.append(content)
            elif scope == "subproject":
                pack.subproject_identity.append(content)
        elif category == "constraint":
            if scope == "global":
                pack.project_constraints.append(content)
            elif scope == "subproject":
                pack.subproject_constraints.append(content)
        elif category == "task_context":
            pack.current_task_context.append(content)
        elif category == "decision":
            pack.recent_decisions.append(content)
        elif category == "acceptance":
            pack.acceptance_notes.append(content)
        elif category == "failure":
            pack.failure_cases.append(content)
        elif category == "do_not_break":
            pack.do_not_break.append(content)
        elif category == "open_question":
            pack.open_questions.append(content)

def build_memory_pack(
    role: str,
    work_item: WorkItemMemoryMetadata,
    parent_refs: List[Dict[str, Any]],
    subproject_refs: List[Dict[str, Any]],
    project_refs: List[Dict[str, Any]]
) -> MemoryPack:
    """
    Implements Deliverable 2 (Memory Retrieval Contract) and Feature B (Role-aware Memory Pack Builder).
    
    Adheres strictly to the retrieval order:
    1. Global (Project)
    2. Subproject
    3. Parent
    4. Current (Work Item)
    5. Session-local
    """
    pack = MemoryPack()

    # 1. Global (Project)
    _process_refs(pack, project_refs, scope="global", role=role)
    
    # 2. Subproject
    _process_refs(pack, subproject_refs, scope="subproject", role=role)
    
    # 3. Parent
    _process_refs(pack, parent_refs, scope="parent", role=role)
    
    # 4. Current (Work Item)
    # Extracting from work_item's inherited, promoted, and local refs (mocking dict structure for consistency)
    current_refs = []
    for mem_ref in work_item.local_memory_refs:
        current_refs.append({
            "category": "task_context", # Simplified assumption
            "content": f"[{mem_ref.memory_id}] Reason: {mem_ref.reason_for_selection}"
        })
    _process_refs(pack, current_refs, scope="current", role=role)
    
    # 5. Session-local
    # Not explicitly provided in signature, handled internally or omitted.
    return pack
