import json
kb = json.load(open("output/knowledge_base.json", encoding="utf-8"))
unassigned = [n for n in kb if n.get("objective_id") is None]
print(f"Total unassigned nodes: {len(unassigned)}")
for n in unassigned:
    t = (n.get("title") or "")[:50]
    print(f"{n['id']} | type={n.get('type'):<20} | parent={n.get('parent_id')} | title={t}")
