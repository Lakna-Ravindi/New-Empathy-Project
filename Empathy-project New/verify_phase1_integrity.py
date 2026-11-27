import json
from pathlib import Path
from collections import Counter, defaultdict

base_path = Path(".")
kb_file = base_path / "output" / "knowledge_base.json"
ps_file = base_path / "output" / "progress_structure.json"
cs_file = base_path / "output" / "content_skills.json"
km_file = base_path / "output" / "keyword_skill_map.json"
vr_file = base_path / "output" / "validation_report.json"

kb = json.load(kb_file.open(encoding="utf-8"))
ps = json.load(ps_file.open(encoding="utf-8"))
cs = json.load(cs_file.open(encoding="utf-8"))
km = json.load(km_file.open(encoding="utf-8"))
vr = json.load(vr_file.open(encoding="utf-8"))

print("=" * 70)
print("PHASE 1: CURRICULUM INTEGRITY VERIFICATION")
print("=" * 70)

# 1. Total nodes & ID uniqueness
node_ids = [n["id"] for n in kb if isinstance(n, dict) and "id" in n]
print(f"Total nodes in knowledge_base.json: {len(kb)}")
print(f"Total unique IDs: {len(set(node_ids))}")
dup_ids = [item for item, count in Counter(node_ids).items() if count > 1]
print(f"Duplicate IDs: {dup_ids if dup_ids else 'None (0)'}")

# Check ID format and continuity between node_0001 and node_0495
expected_range = [f"node_{i:04d}" for i in range(1, 496)]
missing_in_range = [nid for nid in expected_range if nid not in set(node_ids)]
print(f"IDs missing in range node_0001 to node_0495 ({len(missing_in_range)}): {missing_in_range}")
print(f"IDs count: 495 - 3 missing = {495 - len(missing_in_range)}")

# 2. Node types breakdown
types = Counter(n.get("type") for n in kb)
print(f"\nNode types distribution: {dict(types)}")

# 3. Skills in content_skills vs progress_structure vs knowledge_base
cs_skills = {s["skill_id"]: s.get("skill_title") for s in cs}
ps_chapters = {ch["chapter_id"]: ch.get("title") for ch in ps.get("chapters", [])}
kb_chapters = {n["id"]: n.get("title") for n in kb if n.get("type") == "chapter"}
print(f"\nContent skills count: {len(cs_skills)}: {list(cs_skills.keys())}")
print(f"Progress structure chapters count: {len(ps_chapters)}: {list(ps_chapters.keys())}")
print(f"Knowledge base chapter nodes count: {len(kb_chapters)}: {list(kb_chapters.keys())}")

# 4. Objectives verification across all 8 skills
obj_in_ps = []
for ch in ps.get("chapters", []):
    ch_id = ch["chapter_id"]
    for obj in ch.get("objectives", []):
        obj_in_ps.append((ch_id, obj["objective_id"], obj.get("title")))

print(f"\nTotal canonical objectives in progress_structure: {len(obj_in_ps)}")
objs_by_skill = Counter(ch_id for ch_id, _, _ in obj_in_ps)
print("Objectives count per skill:")
for sk in sorted(objs_by_skill.keys()):
    print(f"  {sk}: {objs_by_skill[sk]} objectives")

# Check all 46 objective IDs in knowledge_base
nodes_by_id = {n["id"]: n for n in kb}
kb_objs = set(n.get("objective_id") for n in kb if n.get("objective_id"))
ps_obj_ids = set(oid for _, oid, _ in obj_in_ps)
print(f"\nUnique objective IDs assigned in KB: {len(kb_objs)}")
print(f"PS objectives not in KB: {ps_obj_ids - kb_objs}")
print(f"KB objectives not in PS: {kb_objs - ps_obj_ids}")

# 5. Combined objective headings (Skills 6 and 7)
print("\nCombined objective headings in KB:")
comb_heads = [n for n in kb if n.get("type") == "objective_heading"]
for h in comb_heads:
    print(f"  [{h['id']}] {h.get('objective_id')}: {repr(h.get('title'))}")

# 6. Check node parent references
orphans = []
for n in kb:
    pid = n.get("parent_id")
    if pid and pid not in nodes_by_id:
        orphans.append((n["id"], pid))
print(f"\nOrphan nodes (invalid parent_id): {len(orphans)}")

# 7. Review, unassigned, and not_applicable nodes
unassigned = [n["id"] for n in kb if n.get("objective_id") is None]
print(f"Nodes without objective_id: {len(unassigned)}")
unassigned_types = Counter(nodes_by_id[nid].get("type") for nid in unassigned)
print(f"Types of nodes without objective_id: {dict(unassigned_types)}")

# 8. Backward compatibility: activities, reflections, quizzes
total_items = 0
items_by_type = Counter()
for ch in ps.get("chapters", []):
    for obj in ch.get("objectives", []):
        for it in obj.get("items", []):
            total_items += 1
            items_by_type[it.get("type")] += 1

print(f"\nInteractive items in progress_structure: {total_items}")
print(f"Items by type: {dict(items_by_type)}")
