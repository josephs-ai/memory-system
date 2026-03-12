# Core memory scripts

## Routing / policy
- memory_routing_rules.py
- route_memory_items_batch.py
- route_memory_item.py

## Judgment / extraction
- judge_memory_candidates.py
- extract_memory_fields.py
- dehydrate_transcript.py
- chunk_by_topic.py
- extract_chunk_updates.py

## Storage / canonical
- memory_db.py
- approve_pending_memory_item.py
- sync_registry_to_markdown.py

## Retrieval
- embed_memory_items.py
- search_memory.py
- search_memory_rbac.py

## Notes
- stable_safe_auto policy should be defined in memory_routing_rules.py
- split control panel in .memory-index/control_panel/ is the real UI
- ignore legacy/backup single-file control panel code
