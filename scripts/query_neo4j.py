"""
Query Neo4j — utility for the OpenClaw memory system.

Key functions: main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from graph_store_neo4j import get_neo4j_driver



def _run_lookup(session, *, entity: str | None, prop: str | None, scope_id: str | None):
    if entity and prop:
        return session.run(
            """
            MATCH (e:Entity {name: $entity})-[:HAS_PROPERTY_IN_SCOPE]->(p:Property {key: $property})-[:HAS_VALUE_IN_SCOPE]->(v:Value)
            OPTIONAL MATCH (e)<-[:CONTAINS]-(s:ContextScope {scope_id: e.scope_id})
            WHERE $scope_id IS NULL OR e.scope_id = $scope_id
            RETURN e.name AS entity, p.key AS property, v.key AS value, e.scope_id AS scope_id, s.scope_type AS scope_type
            ORDER BY scope_id, value
            """,
            entity=entity,
            property=prop,
            scope_id=scope_id,
        )
    if entity:
        return session.run(
            """
            MATCH (e:Entity {name: $entity})-[:HAS_PROPERTY_IN_SCOPE]->(p:Property)-[:HAS_VALUE_IN_SCOPE]->(v:Value)
            OPTIONAL MATCH (e)<-[:CONTAINS]-(s:ContextScope {scope_id: e.scope_id})
            WHERE $scope_id IS NULL OR e.scope_id = $scope_id
            RETURN e.name AS entity, p.key AS property, v.key AS value, e.scope_id AS scope_id, s.scope_type AS scope_type
            ORDER BY scope_id, property, value
            """,
            entity=entity,
            scope_id=scope_id,
        )
    return session.run(
        """
        MATCH (e:Entity)-[:HAS_PROPERTY_IN_SCOPE]->(p:Property)-[:HAS_VALUE_IN_SCOPE]->(v:Value)
        OPTIONAL MATCH (e)<-[:CONTAINS]-(s:ContextScope {scope_id: e.scope_id})
        WHERE $scope_id IS NULL OR e.scope_id = $scope_id
        RETURN e.name AS entity, p.key AS property, v.key AS value, e.scope_id AS scope_id, s.scope_type AS scope_type
        ORDER BY scope_id, entity, property, value
        LIMIT 20
        """,
        scope_id=scope_id,
    )



def _run_text_search(session, *, query: str, limit: int, project_id: str | None, subproject_id: str | None, workflow_id: str | None, pipeline_id: str | None):
    return session.run(
        """
        MATCH (m:MemoryItem)-[:ASSERTED_IN]->(s:ContextScope)
        WHERE toLower(m.text) CONTAINS toLower($query)
          AND ($project_id IS NULL OR s.project_id = $project_id)
          AND ($subproject_id IS NULL OR s.subproject_id = $subproject_id)
          AND ($workflow_id IS NULL OR s.workflow_id = $workflow_id)
          AND ($pipeline_id IS NULL OR s.pipeline_id = $pipeline_id)
        RETURN
          m.text AS text,
          m.id AS memory_id,
          m.memory_type AS memory_type,
          m.confidence AS score,
          s.scope_id AS scope_id,
          s.scope_type AS scope_type,
          s.project_id AS project_id,
          s.subproject_id AS subproject_id,
          s.workflow_id AS workflow_id,
          s.pipeline_id AS pipeline_id
        ORDER BY m.confidence DESC, m.importance DESC, m.id ASC
        LIMIT $limit
        """,
        # Passed as an explicit parameters dict, not as keyword arguments.
        # Session.run()'s own first parameter is named "query", so a Cypher
        # parameter of the same name collided with it:
        #   TypeError: Session.run() got multiple values for argument 'query'
        # That made every graph-backed search fail while the DB and vector
        # sources still returned results, so the service looked healthy.
        parameters={
            "query": query,
            "limit": limit,
            "project_id": project_id,
            "subproject_id": subproject_id,
            "workflow_id": workflow_id,
            "pipeline_id": pipeline_id,
        },
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity")
    parser.add_argument("--property")
    parser.add_argument("--scope-id")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--project-id")
    parser.add_argument("--subproject-id")
    parser.add_argument("--workflow-id")
    parser.add_argument("--pipeline-id")
    args = parser.parse_args()

    driver = get_neo4j_driver()
    with driver.session() as session:
        if args.query:
            rows = _run_text_search(
                session,
                query=args.query,
                limit=args.limit,
                project_id=args.project_id,
                subproject_id=args.subproject_id,
                workflow_id=args.workflow_id,
                pipeline_id=args.pipeline_id,
            )
            found = False
            for row in rows:
                found = True
                print(
                    json.dumps(
                        {
                            "text": row["text"],
                            "path": f"neo4j://scope/{row['scope_id']}",
                            "source_type": "graph",
                            "score": row["score"],
                            "memory_id": row["memory_id"],
                            "memory_type": row["memory_type"],
                            "scope_id": row["scope_id"],
                            "scope_type": row["scope_type"],
                            "project_id": row["project_id"],
                            "subproject_id": row["subproject_id"],
                            "workflow_id": row["workflow_id"],
                            "pipeline_id": row["pipeline_id"],
                        }
                    )
                )
            if not found:
                print("NONE")
        else:
            rows = _run_lookup(session, entity=args.entity, prop=args.property, scope_id=args.scope_id)
            found = False
            for row in rows:
                found = True
                print(
                    json.dumps(
                        {
                            "entity": row["entity"],
                            "property": row["property"],
                            "value": row["value"],
                            "scope_id": row["scope_id"],
                            "scope_type": row["scope_type"],
                        }
                    )
                )
            if not found:
                print("NONE")

    driver.close()


if __name__ == "__main__":
    main()
