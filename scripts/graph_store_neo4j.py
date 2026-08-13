"""
Neo4j graph database adapter for storing and querying code structure
and memory relationships.
"""
from __future__ import annotations

import json
import os
from typing import Any

from context_scope import build_context_scope

NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")


def get_neo4j_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )



def ensure_memory_constraints() -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("CREATE CONSTRAINT memory_item_id IF NOT EXISTS FOR (m:MemoryItem) REQUIRE m.id IS UNIQUE")
        session.run("DROP CONSTRAINT entity_name IF EXISTS")
        session.run("DROP CONSTRAINT property_key IF EXISTS")
        session.run("DROP CONSTRAINT value_key IF EXISTS")
        session.run("CREATE CONSTRAINT context_scope_id IF NOT EXISTS FOR (s:ContextScope) REQUIRE s.scope_id IS UNIQUE")
        session.run("CREATE CONSTRAINT entity_identity IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.scope_id) IS UNIQUE")
        session.run("CREATE CONSTRAINT property_identity IF NOT EXISTS FOR (p:Property) REQUIRE (p.key, p.scope_id) IS UNIQUE")
        session.run("CREATE CONSTRAINT value_identity IF NOT EXISTS FOR (v:Value) REQUIRE (v.key, v.scope_id) IS UNIQUE")
    driver.close()



def prepare_graph_item(item: dict[str, Any]) -> dict[str, Any] | None:
    memory_id = item.get("id")
    text = item.get("text")
    entity = item.get("entity")
    prop = item.get("property")
    value = item.get("value")
    memory_type = item.get("memory_type")
    confidence = float(item.get("confidence") or 0.0)
    importance = float(item.get("importance") or 0.0)

    if not memory_id or not entity or not prop or value is None:
        return None

    scope_fields = build_context_scope(item)
    return {
        "memory_id": memory_id,
        "text": text,
        "entity": str(entity),
        "prop": str(prop),
        "value": str(value),
        "memory_type": memory_type,
        "confidence": confidence,
        "importance": importance,
        "scope": str(item.get("scope") or scope_fields.get("scope") or "stable"),
        "project_id": scope_fields.get("project_id"),
        "subproject_id": scope_fields.get("subproject_id"),
        "workflow_id": scope_fields.get("workflow_id"),
        "pipeline_id": scope_fields.get("pipeline_id"),
        "scope_id": scope_fields.get("scope_id"),
        "scope_type": scope_fields.get("scope_type"),
        "inheritance_policy": scope_fields.get("inheritance_policy"),
        "scope_confidence": float(scope_fields.get("scope_confidence") or 0.0),
        # Neo4j properties must be primitives or arrays of primitives; a map is
        # rejected outright, so the full scope record is stored as JSON text.
        "scope_payload": json.dumps(dict(scope_fields), sort_keys=True, default=str),
    }



def upsert_memory_graph(item: dict) -> None:
    payload = prepare_graph_item(item)
    if not payload:
        return

    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (s:ContextScope {scope_id: $scope_id})
            SET s.scope_type = $scope_type,
                s.scope = $scope,
                s.project_id = $project_id,
                s.subproject_id = $subproject_id,
                s.workflow_id = $workflow_id,
                s.pipeline_id = $pipeline_id,
                s.inheritance_policy = $inheritance_policy,
                s.scope_confidence = $scope_confidence,
                s.payload = $scope_payload

            MERGE (m:MemoryItem {id: $memory_id})
            SET m.text = $text,
                m.scope = $scope,
                m.memory_type = $memory_type,
                m.confidence = $confidence,
                m.importance = $importance,
                m.context_scope_id = $scope_id,
                m.context_scope_type = $scope_type,
                m.inheritance_policy = $inheritance_policy

            MERGE (e:Entity {name: $entity, scope_id: $scope_id})
            SET e.scope_type = $scope_type,
                e.project_id = $project_id,
                e.subproject_id = $subproject_id,
                e.workflow_id = $workflow_id,
                e.pipeline_id = $pipeline_id

            MERGE (p:Property {key: $prop, scope_id: $scope_id})
            SET p.scope_type = $scope_type

            MERGE (v:Value {key: $value, scope_id: $scope_id})
            SET v.scope_type = $scope_type

            MERGE (m)-[:ASSERTED_IN]->(s)
            MERGE (s)-[:CONTAINS]->(e)
            MERGE (s)-[:CONTAINS]->(p)
            MERGE (s)-[:CONTAINS]->(v)
            MERGE (m)-[:ABOUT_ENTITY]->(e)
            MERGE (m)-[:HAS_PROPERTY]->(p)
            MERGE (m)-[:HAS_VALUE]->(v)
            MERGE (e)-[:HAS_PROPERTY_IN_SCOPE]->(p)
            MERGE (p)-[:HAS_VALUE_IN_SCOPE]->(v)
            """,
            **payload,
        )
    driver.close()
