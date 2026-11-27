import json, sys
from pathlib import Path
backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))
from learning.pedagogical_controller import score_text, expand_terms

kb = json.load(open("output/knowledge_base.json", encoding="utf-8"))
q = "What is the resilient zone and what happens when we are stuck on high?"
terms = expand_terms(q)
for n in kb:
    if n.get("objective_id") == "obj_01_02" and n.get("type") == "content":
        s = score_text(f"{n.get('title','')} {n.get('content','')}", terms)
        if s > 0:
            print(n["id"], s, n["title"][:50])
