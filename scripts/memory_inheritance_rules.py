"""
Rules governing how memory items inherit scope and visibility
from parent projects and workspaces.
"""
from typing import List, Dict, Any

class InheritanceRules:
    """
    Deliverable 3: Inheritance Rules
    Feature C: Planner child memory assignment logic.
    """
    def __init__(self):
        pass

    def assign_child_memory(self, parent_memory: List[Dict[str, Any]], child_scope: str, task_requirements: List[str]) -> List[Dict[str, Any]]:
        """
        Assigns relevant memory from the parent context to the child context based on scope and task requirements.
        This is an additive improvement, preserving original parent state while seeding the child planner.
        """
        child_memory = []

        for item in parent_memory:
            item_scope = item.get("scope", "global")
            item_tags = item.get("tags", [])

            # Rule 1: Always inherit global durable memory
            if item_scope == "global" and item.get("category") == "Durable":
                child_memory.append(self._clone_item_for_child(item))
                continue

            # Rule 2: Inherit memory specifically scoped to this child or matching task requirements
            if item_scope == child_scope or any(req in item_tags for req in task_requirements):
                child_memory.append(self._clone_item_for_child(item))
                continue

        return child_memory

    def _clone_item_for_child(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely clone memory item for child to prevent unintended mutations of parent state.
        """
        cloned_item = item.copy()
        cloned_item["inherited"] = True
        return cloned_item
