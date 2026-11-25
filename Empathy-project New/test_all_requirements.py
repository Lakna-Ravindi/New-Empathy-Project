import json, sys
from pathlib import Path

backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))
from learning.pedagogical_controller import process_student_question, curriculum_skill_id

kb = json.load(open("output/knowledge_base.json", encoding="utf-8"))
ps = json.load(open("output/progress_structure.json", encoding="utf-8"))
nodes_by_id = {n["id"]: n for n in kb if isinstance(n, dict) and "id" in n}

print("=" * 70)
print("TEST 1: 5 SPECIFIC TARGET QUERIES (Q05, Q09, Q13, Q15, Q19)")
print("=" * 70)

target_tests = [
    ("Q05", "skill_02", "obj_02_03", "How does the playful puppy metaphor explain stabilizing my wandering mind during study?"),
    ("Q09", "skill_03", "obj_03_03", "How does selective filtering create space between an impulse and reacting impulsively?"),
    ("Q13", "skill_05", "obj_05_01", "What do all human beings share in common despite surface cultural and social differences?"),
    ("Q15", "skill_06", "obj_06_01", "Why is forgiveness considered a demonstration of inner strength rather than weakness or condoning harm?"),
    ("Q19", "skill_08", "obj_08_03", "What are the three necessary components of compassion in SEEK: noticing suffering, empathic concern, and agency?")
]

t1_passed = 0
for qid, exp_s, exp_o, text in target_tests:
    res = process_student_question(text, kb, progress_structure=ps)
    s = res.get("skill") or {}
    act_s = curriculum_skill_id(s.get("id"), kb, nodes_by_id) if s else None
    obj = res.get("learning_objective") or {}
    act_o = obj.get("objective_id") if obj else None
    status = res.get("status")
    nodes = [n.get("id") for n in res.get("question_relevant_content", [])]
    ok = (act_s == exp_s and act_o == exp_o and status == "learning_path_selected")
    if ok:
        t1_passed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {qid}: exp {exp_s}/{exp_o} -> got {act_s}/{act_o} | nodes={nodes[:3]}")

print(f"\nTarget queries result: {t1_passed}/{len(target_tests)} passed.\n")

print("=" * 70)
print("TEST 2: CONVERSATIONAL PARAPHRASE TESTS")
print("=" * 70)

paraphrases = [
    ("Puppy 1", "skill_02", "obj_02_03", "My mind is like a playful puppy wandering around when I study."),
    ("Puppy 2", "skill_02", "obj_02_03", "How do I deal with a wandering mind using the puppy training idea?"),
    ("Filtering 1", "skill_03", "obj_03_03", "How can selective filtering help me not react impulsively?"),
    ("Filtering 2", "skill_03", "obj_03_03", "Can selective filtering create space between an impulse and my reaction?"),
    ("Common humanity 1", "skill_05", "obj_05_01", "What do human beings share in common according to common humanity?"),
    ("Common humanity 2", "skill_05", "obj_05_01", "Why do all humans experience suffering and share common experiences?"),
    ("Forgiveness 1", "skill_06", "obj_06_01", "Is forgiving someone a sign of weakness or giving in to harm?"),
    ("Forgiveness 2", "skill_06", "obj_06_01", "How is forgiveness an inner strength rather than condoning bad actions?"),
    ("Compassion parts 1", "skill_08", "obj_08_03", "What are the three components of compassion: noticing suffering, empathic concern, and agency?"),
    ("Compassion levels 1", "skill_08", "obj_08_04", "What are the three levels of compassion: wishing, aspiring, and engaged?")
]

t2_passed = 0
for name, exp_s, exp_o, text in paraphrases:
    res = process_student_question(text, kb, progress_structure=ps)
    s = res.get("skill") or {}
    act_s = curriculum_skill_id(s.get("id"), kb, nodes_by_id) if s else None
    obj = res.get("learning_objective") or {}
    act_o = obj.get("objective_id") if obj else None
    ok = (act_s == exp_s and (exp_o is None or act_o == exp_o))
    if ok:
        t2_passed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: exp {exp_s}/{exp_o} -> got {act_s}/{act_o}")

print(f"\nParaphrase queries result: {t2_passed}/{len(paraphrases)} passed.\n")

print("=" * 70)
print("TEST 3: CROSS-SKILL DISAMBIGUATION TESTS")
print("=" * 70)

cross_skill_tests = [
    ("Empathy vs Compassion", "skill_07", "obj_07_02", "What is the difference between cognitive and affective empathy?"),
    ("Empathic concern in empathy", "skill_07", None, "How does empathic concern develop in emotional connection?"),
    ("Compassion with empathic concern", "skill_08", "obj_08_03", "What are the three necessary components of compassion including noticing suffering and empathic concern?"),
    ("Forgiveness with harm/strength", "skill_06", "obj_06_01", "Why does forgiveness require inner strength when someone caused harm?"),
    ("Self-compassion without forgiveness", "skill_04", None, "How does self-compassion help me overcome self-criticism?"),
]

t3_passed = 0
for name, exp_s, exp_o, text in cross_skill_tests:
    res = process_student_question(text, kb, progress_structure=ps)
    s = res.get("skill") or {}
    act_s = curriculum_skill_id(s.get("id"), kb, nodes_by_id) if s else None
    obj = res.get("learning_objective") or {}
    act_o = obj.get("objective_id") if obj else None
    ok = (act_s == exp_s and (exp_o is None or act_o == exp_o))
    if ok:
        t3_passed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: exp {exp_s}/{exp_o} -> got {act_s}/{act_o}")

print(f"\nCross-skill queries result: {t3_passed}/{len(cross_skill_tests)} passed.\n")

print("=" * 70)
print("TEST 4: COMPOUND-QUESTION MULTI-OBJECTIVE CONTENT RETRIEVAL")
print("=" * 70)

compound_q = "What is the resilient zone and what happens when we are stuck on high?"
res_cq = process_student_question(compound_q, kb, progress_structure=ps)
primary_obj = (res_cq.get("learning_objective") or {}).get("objective_id")
retrieved_nodes = res_cq.get("question_relevant_content", [])
retrieved_objs = set()
for r in retrieved_nodes:
    node = nodes_by_id.get(r.get("id"))
    if node and node.get("objective_id"):
        retrieved_objs.add(node["objective_id"])

print(f"Compound question: '{compound_q}'")
print(f"Primary objective: {primary_obj}")
print(f"Retrieved node count: {len(retrieved_nodes)}")
print(f"Objectives represented in retrieved nodes: {retrieved_objs}")
has_multiple_objs = len(retrieved_objs) > 1
print(f"Compound multi-objective retrieval: {'PASSED' if has_multiple_objs else 'SINGLE_OBJ'}")
