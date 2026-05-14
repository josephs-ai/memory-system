from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.db import get_conn


def test_retry_blocked_work_item_resets_state():
    work_id = str(uuid.uuid4())

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into work_items (
                work_id,
                project_id,
                subproject_id,
                title,
                description,
                kind,
                status,
                priority,
                risk_level,
                memory_mode_effective,
                next_action,
                assigned_role,
                requires_review,
                requires_test,
                retry_count,
                max_retries,
                blocked_reason,
                depends_on_work_id,
                requested_by,
                approval_state,
                auto_approve_allowed,
                escalation_reason,
                last_error_summary
            )
            values (
                %s, %s, null,
                %s,
                %s,
                %s,
                'blocked',
                50,
                'medium',
                'game_dev',
                'wait',
                null,
                true,
                true,
                2,
                3,
                'Review retry limit reached',
                null,
                'test',
                'not_required',
                true,
                'review_retry_limit',
                'old error'
            )
            """,
            (
                work_id,
                "test_project",
                "Blocked item test",
                "Retry blocked integration test",
                "feature",
            ),
        )
        conn.commit()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "orchestrator.retry_blocked_work_item",
            "--work-id",
            work_id,
            "--reset-retry-count",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select status, next_action, assigned_role, retry_count,
                   blocked_reason, escalation_reason, last_error_summary
            from work_items
            where work_id = %s
            """,
            (work_id,),
        )
        row = cur.fetchone()

    assert row == ("queued", "build", None, 0, None, None, None)
