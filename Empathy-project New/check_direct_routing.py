import json, sys
from pathlib import Path

backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))
from learning.pedagogical_controller import process_student_question, _direct_skill_groups, curriculum_skill_id

kb = json.load(open("output/knowledge_base.json", encoding="utf-8"))
ps = json.load(open("output/progress_structure.json", encoding="utf-8"))
nodes_by_id = {n["id"]: n for n in kb if isinstance(n, dict) and "id" in n}

questions = [
    ("Q05", "skill_02", "obj_02_03", "How does the playful puppy metaphor explain stabilizing my wandering mind during study?"),
    ("Q09", "skill_03", "obj_03_03", "How does selective filtering create space between an impulse and reacting impulsively?"),
    ("Q13", "skill_05", "obj_05_01", "What do all human beings share in common despite surface cultural and social differences?"),
    ("Q15", "skill_06", "obj_06_01", "Why is forgiveness considered a demonstration of inner strength rather than weakness or condoning harm?"),
    ("Q19", "skill_08", "obj_08_03", "What are the three necessary components of compassion in SEEK: noticing suffering, empathic concern, and agency?")
]

print("=== Direct process_student_question (without highlighted_keywords) ===")
for qid, exp_s, exp_o, text in questions:
    res = process_student_question(text, kb, progress_structure=ps)
    s = res.get("skill") or {}
    actual_s = curriculum_skill_id(s.get("id"), kb, nodes_by_id) if s else None
    obj = res.get("learning_objective") or {}
    actual_o = obj.get("objective_id") if obj else None
    print(f"{qid}: expected {exp_s}/{exp_o} -> got {actual_s}/{actual_o} (status={res.get('status')})")

from classifier.rule_classifier import match_intent
print("\n=== process_student_question with rule_classifier match_intent ===")
for qid, exp_s, exp_o, text in questions:
    hk = match_intent(text)
    res = process_student_question(text, kb, highlighted_keywords=hk, progress_structure=ps)
    s = res.get("skill") or {}
    actual_s = curriculum_skill_id(s.get("id"), kb, nodes_by_id) if s else None
    obj = res.get("learning_objective") or {}
    actual_o = obj.get("objective_id") if obj else None
    status = res.get("status")
    print(f"{qid}: expected {exp_s}/{exp_o} -> got {actual_s}/{actual_o} (status={status})")

