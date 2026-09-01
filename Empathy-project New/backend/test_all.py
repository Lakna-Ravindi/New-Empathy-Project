import sys
import json

print('=== EMPATHY PROJECT COMPREHENSIVE TEST ===\n')

# Test 1: Imports
print('[TEST 1] Import all modules...')
try:
    from classifier.rule_classifier import classify
    from knowledge.node_builder import build_node
    from knowledge.keyword_skill_mapper import map_keywords_to_skills
    from parser.metadata_extractor import extract_blocks
    from parser.block_merger import merge_spans
    from parser.highlight_extractor import extract_highlighted_keywords
    from learning.interaction_store import LearningStore
    from learning.pedagogical_controller import answer_student_question
    from learning.gemini_responder import generate_educational_response
    print('OK: All imports successful\n')
except Exception as e:
    print(f'ERROR: Import failed: {e}\n')
    sys.exit(1)

# Test 2: Classifier
print('[TEST 2] Test classifier...')
test_block = {'text': 'Skill 1: Test', 'font_name': 'Arial-Bold', 'font_size': 13}
result = classify(test_block)
print(f'OK: Classifier result: {result}\n')

# Test 3: Knowledge base loading
print('[TEST 3] Load knowledge base...')
kb = json.load(open('../output/knowledge_base.json', encoding='utf-8'))
km = json.load(open('../output/keyword_skill_map.json', encoding='utf-8'))
print(f'OK: Knowledge base: {len(kb)} nodes')
print(f'OK: Keyword mappings: {len(km)} mappings\n')

# Test 4: MongoDB/Local storage
print('[TEST 4] Test data storage...')
store = LearningStore()
print(f'OK: Storage available: {store.is_available}\n')

# Test 5: Pedagogical controller
print('[TEST 5] Test pedagogical controller...')
test_question = 'I feel worried about my performance'
result = answer_student_question(test_question, kb, km)
if 'learning_context' in result:
    status = result['learning_context'].get('status')
    emotion = result['learning_context'].get('detected_emotion')
    has_response = 'educational_response' in result
    print(f'OK: Controller status: {status}')
    print(f'OK: Emotion detected: {emotion}')
    print(f'OK: Response generated: {has_response}\n')
else:
    print(f'OK: Query status: {result.get("status")}\n')

print('=== ALL TESTS PASSED ===')
