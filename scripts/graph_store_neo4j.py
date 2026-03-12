from __future__ import annotations

import os
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")


def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )


def ensure_memory_constraints() -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("CREATE CONSTRAINT memory_item_id IF NOT EXISTS FOR (m:MemoryItem) REQUIRE m.id IS UNIQUE")
        session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        session.run("CREATE CONSTRAINT property_key IF NOT EXISTS FOR (p:Property) REQUIRE p.key IS UNIQUE")
        session.run("CREATE CONSTRAINT value_key IF NOT EXISTS FOR (v:Value) REQUIRE v.key IS UNIQUE")
    driver.close()


def upsert_memory_graph(item: dict) -> None:
    memory_id = item.get("id")
    text = item.get("text")
    entity = item.get("entity")
    prop = item.get("property")
    value = item.get("value")
    scope = item.get("scope")
    memory_type = item.get("memory_type")
    confidence = item.get("confidence")
    importance = item.get("importance")

    if not memory_id or not entity or not prop or value is None:
        return

    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (m:MemoryItem {id: $memory_id})
            SET m.text = $text,
                m.scope = $scope,
                m.memory_type = $memory_type,
                m.confidence = $confidence,
                m.importance = $importance

            MERGE (e:Entity {name: $entity})
            MERGE (p:Property {key: $prop})
            MERGE (v:Value {key: $value})

            MERGE (m)-[:ABOUT_ENTITY]->(e)
            MERGE (m)-[:HAS_PROPERTY]->(p)
            MERGE (m)-[:HAS_VALUE]->(v)

            MERGE (e)-[:HAS_PROPERTY]->(p)
            MERGE (p)-[:HAS_VALUE]->(v)
            """,
            memory_id=memory_id,
            text=text,
            entity=entity,
            prop=prop,
            value=str(value),
            scope=scope,
            memory_type=memory_type,
            confidence=confidence,
            importance=importance,
        )
    driver.close()
