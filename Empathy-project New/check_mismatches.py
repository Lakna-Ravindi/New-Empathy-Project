import json, sys
from pathlib import Path

backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))
from learning.pedagogical_controller import score_text, expand_terms, rank_learning_objectives

kb = json.load(open("output/knowledge_base.json", encoding="utf-8"))
ps = json.load(open("output/progress_structure.json", encoding="utf-8"))
nodes_by_id = {n["id"]: n for n in kb if isinstance(n, dict) and "id" in n}

test_cases = [
    ("Q01", "skill_01", "node_0004", "How can tracking physical body sensations help me recognize when I am starting to get angry?"),
    ("Q04", "skill_02", "node_0091", "What is the difference between heedfulness, mindfulness, and awareness in ethical mindfulness?"),
    ("Q06", "skill_02", "node_0091", "How can ethical mindfulness help me make value-aligned choices when pressured by peers?"),
    ("Q10", "skill_04", "node_0213", "What is self-compassion and why is treating myself kindly not the same as being selfish?")
]

for qid, sk, ch_id, q in test_cases:
    terms = expand_terms(q)
    ranked = rank_learning_objectives(ch_id, kb, nodes_by_id, terms, progress_structure=ps)
    print(f"\n--- {qid} ---")
    for o in ranked[:3]:
        print(f"  {o.get('objective_id')}: {o.get('title')}")
    print("Content nodes for top candidate objectives:")
    for n in kb:
        if n.get("objective_id") in [o.get("objective_id") for o in ranked[:2]] and n.get("type") == "content":
            s = score_text(f"{n.get('title', '')} {n.get('content', '')}", terms)
            if s > 1:
                print(f"    [{n.get('objective_id')}] {n.get('id')} (score={s}): {n.get('title')}")

