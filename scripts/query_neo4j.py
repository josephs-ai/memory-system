from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from graph_store_neo4j import get_neo4j_driver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity")
    parser.add_argument("--property")
    args = parser.parse_args()

    driver = get_neo4j_driver()
    with driver.session() as session:
        if args.entity and args.property:
            rows = session.run(
                """
                MATCH (e:Entity {name: $entity})-[:HAS_PROPERTY]->(p:Property {key: $property})-[:HAS_VALUE]->(v:Value)
                RETURN e.name AS entity, p.key AS property, v.key AS value
                ORDER BY value
                """,
                entity=args.entity,
                property=args.property,
            )
        elif args.entity:
            rows = session.run(
                """
                MATCH (e:Entity {name: $entity})-[:HAS_PROPERTY]->(p:Property)-[:HAS_VALUE]->(v:Value)
                RETURN e.name AS entity, p.key AS property, v.key AS value
                ORDER BY property, value
                """,
                entity=args.entity,
            )
        else:
            rows = session.run(
                """
                MATCH (e:Entity)-[:HAS_PROPERTY]->(p:Property)-[:HAS_VALUE]->(v:Value)
                RETURN e.name AS entity, p.key AS property, v.key AS value
                ORDER BY entity, property, value
                LIMIT 20
                """
            )

        found = False
        for row in rows:
            found = True
            print(
                {
                    "entity": row["entity"],
                    "property": row["property"],
                    "value": row["value"],
                }
            )

        if not found:
            print("NONE")

    driver.close()


if __name__ == "__main__":
    main()
