from t6_e2e import fetch_search, datetime, timezone
AS_OF = datetime(2026, 6, 18, 23, 0, tzinfo=timezone.utc)
rows_leg = fetch_search("user", weight=0.0, half_life=14.0, as_of=AS_OF, limit=5)
rows_rec = fetch_search("user", weight=0.6, half_life=14.0, as_of=AS_OF, limit=5)
print("--- LEGACY (weight=0) ---")
for r in rows_leg:
    print(f"ID {r['id'][:8]} | recency_bonus={r['abs_recency_bonus']} | final_score={r['final_score']}")
print("\n--- RECENCY (weight=0.6) ---")
for r in rows_rec:
    print(f"ID {r['id'][:8]} | recency_bonus={round(r['abs_recency_bonus'],4)} | final_score={round(r['final_score'],4)}")
