import sys
sys.path.insert(0, "backend")
from agents.tools.curriculum_tools import get_learning_activity

print("Checking get_learning_activity across canonical objectives...")
for s_num in range(1, 9):
    sk_id = f"skill_{s_num:02d}"
    obj_id = f"obj_{s_num:02d}_01"
    try:
        act = get_learning_activity(skill_id=sk_id, objective_id=obj_id)
        print(f"  {sk_id} / {obj_id} -> id={act.get('id')}, type={act.get('type')}, title={repr(act.get('title')[:35])}")
    except Exception as e:
        print(f"  {sk_id} / {obj_id} -> ERROR: {e}")
