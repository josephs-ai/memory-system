from __future__ import annotations

from orchestrator.repository import create_child_work_items_from_plan
from orchestrator.db import get_conn
import uuid


def test_system_designer_inserted_before_technical_designer_and_builder_chain() -> None:
    parent_work_id = str(uuid.uuid4())

    planner_result = {
        "accepted_specialist_roles": ["system_designer", "technical_designer"],
        "proposed_child_items": [
            {
                "child_key": "child_1",
                "title": "Gameplay framework slice",
                "description": "Build first slice",
                "kind": "feature",
                "priority": 50,
                "risk_level": "medium",
                "recommended_role": "builder",
                "tags": ["cross_system"],
                "memory_refs": [],
                "depends_on_child_key": None,
            },
            {
                "child_key": "child_2",
                "title": "Gameplay framework integration slice",
                "description": "Build second slice",
                "kind": "feature",
                "priority": 50,
                "risk_level": "medium",
                "recommended_role": "builder",
                "tags": ["cross_system"],
                "memory_refs": [],
                "depends_on_child_key": "child_1",
            },
        ],
    }

    create_child_work_items_from_plan(
        parent_work_id=parent_work_id,
        project_id="test_project",
        subproject_id=None,
        requested_by="planner",
        planner_result=planner_result,
    )

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT title, next_action, depends_on_work_id
            FROM work_items
            WHERE parent_work_id = %s
            ORDER BY created_at ASC
        """, (parent_work_id,))
        rows = cur.fetchall()

    assert len(rows) == 4
    assert rows[0][1] == "system_design"
    assert rows[1][1] == "technical_design"
    assert rows[1][2] is not None
    assert rows[2][1] == "build"
    assert rows[2][2] is not None
