from __future__ import annotations

import argparse
import json

from checkpoint_db import (
    close_pool,
    ensure_checkpoint_tables,
    get_agent_last_loaded_checkpoint,
    latest_committed_checkpoint_id,
    mark_agent_loaded,
)

from search_runtime import run_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--project-id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        ensure_checkpoint_tables()

        latest_id = latest_committed_checkpoint_id()
        last_loaded = get_agent_last_loaded_checkpoint(args.agent)

        if not args.force and (latest_id is None or latest_id == last_loaded):
            print("NONE")
            return

        rows = run_search(
            query=args.query,
            top_k=args.top_k,
            project_id=args.project_id,
        )

        if latest_id is not None:
            mark_agent_loaded(args.agent, latest_id)

        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    finally:
        close_pool()
