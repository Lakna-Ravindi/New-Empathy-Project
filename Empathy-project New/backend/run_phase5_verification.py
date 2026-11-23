"""Phase 5 Comprehensive End-to-End Verification Suite for General Empathy Agent.

Validates:
1. Application startup, imports, and endpoint mapping.
2. 7 distinct conversation scenarios (English, Sinhala, Singlish, Academic stress,
   Group project conflict, No-advice request, Multi-turn context).
3. Safety screening gate and crisis detection.
4. Security, JWT authentication, and identity enforcement.
5. Database integrity (checking collections, progress immutability, and SEEK separation).
"""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import auth
import server
from agents.definitions.general_empathy_agent import (
    GeneralEmpathyAgent,
    GeneralEmpathyInput,
    GeneralEmpathyOutput,
)


def log_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def run_e2e_verification():
    results = {
        "startup": {},
        "conversations": [],
        "safety": [],
        "backend_security": [],
        "database_integrity": {},
        "seek_preservation": {},
    }

    # -----------------------------------------------------------------------
    # 1. Application Startup & Route Verification
    # -----------------------------------------------------------------------
    log_section("1. APPLICATION STARTUP & ROUTE VERIFICATION")
    routes = [r.rule for r in server.app.url_map.iter_rules()]
    has_general_chat = "/api/general-empathy-chat" in routes
    has_learning_resp = "/api/learning-response" in routes

    print(f"Flask App loaded: {server.app.name}")
    print(f"General Empathy Endpoint Present: {has_general_chat}")
    print(f"SEEK Learning Endpoint Present: {has_learning_resp}")
    print(f"Orchestrator instantiated: {server.orchestrator is not None}")
    print(f"GeneralEmpathyAgent instantiated: {server.general_empathy_agent is not None}")
    print(f"Agent role: {server.general_empathy_agent.role}")
    print(f"Agent fallback_on_error: {server.general_empathy_agent.fallback_on_error}")

    results["startup"] = {
        "status": "PASS" if (has_general_chat and has_learning_resp) else "FAIL",
        "endpoints": [r for r in routes if "empathy" in r or "learning" in r],
    }

    # -----------------------------------------------------------------------
    # 2. Database Pre-Verification Snapshot
    # -----------------------------------------------------------------------
    log_section("2. DATABASE PRE-VERIFICATION SNAPSHOT")
    db = server.student_store.db if server.student_store else None
    pre_collections = []
    pre_progress_count = 0
    pre_interactions_count = 0

    existing_student = None
    if db is not None:
        try:
            pre_collections = sorted(db.list_collection_names())
            pre_progress_count = db["student_progress"].count_documents({})
            pre_interactions_count = db["learning_interactions"].count_documents({})
            print(f"Connected MongoDB Database: {db.name}")
            print(f"Existing Collections ({len(pre_collections)}): {pre_collections}")
            print(f"Initial student_progress count: {pre_progress_count}")
            print(f"Initial learning_interactions count: {pre_interactions_count}")

            existing_student = db["users"].find_one({"role": "student"})
        except Exception as e:
            print(f"MongoDB snapshot warning: {e}")

    if not existing_student:
        existing_student = {
            "_id": "6a97d0e959b45adb6a0c8c45",
            "username": "test_user",
            "role": "student",
            "student_id": "6a97d0e959b45adb6a0c8c45",
        }

    token = auth.create_access_token(existing_student)
    auth_headers = {"Authorization": f"Bearer {token}"}
    client = server.app.test_client()

    print(f"Authenticated as student: {existing_student.get('username')} (ID: {existing_student.get('_id')})")

    # -----------------------------------------------------------------------
    # 3. Frontend Conversation Tests (Live Agent Execution)
    # -----------------------------------------------------------------------
    log_section("3. CONVERSATION SCENARIO TESTS (7 LIVE RUNS)")

    test_scenarios = [
        {
            "id": 1,
            "name": "Normal English Conversation",
            "message": "I had a really tough day today, everything seemed to go wrong from the moment I woke up.",
            "history": [],
            "language_check": "en",
        },
        {
            "id": 2,
            "name": "Sinhala Conversation",
            "message": "මට අද හරිම මහන්සියි සහ කලකිරිල ඉන්නේ. කිසිම වැඩක් හරියට කරගන්න බැරි වුණා.",
            "history": [],
            "language_check": "si",
        },
        {
            "id": 3,
            "name": "Singlish Conversation",
            "message": "Mata ada hariyata stress wela inne exam nisa. Padam karannath hithenne na.",
            "history": [],
            "language_check": "singlish",
        },
        {
            "id": 4,
            "name": "Academic Stress Expression",
            "message": "I have 3 exams next week and an assignment due tomorrow, and I feel completely overwhelmed.",
            "history": [],
            "language_check": "en",
        },
        {
            "id": 5,
            "name": "University Group Project Problem",
            "message": "My university group project members aren't contributing anything, and the final submission is in two days. I feel so frustrated carrying all this alone.",
            "history": [],
            "language_check": "en",
        },
        {
            "id": 6,
            "name": "Explicit No-Advice Request",
            "message": "Please don't give me any advice, tips, or breathing exercises. I just want to vent because I'm so annoyed with how my day went.",
            "history": [],
            "language_check": "en",
            "no_advice_check": True,
        },
        {
            "id": 7,
            "name": "Multi-Turn Conversational Continuity",
            "message": "She completely ignored my messages afterwards, which made me feel even worse.",
            "history": [
                {"role": "user", "content": "I had a painful misunderstanding with my closest friend this morning."},
                {"role": "assistant", "content": "Misunderstandings with someone you care about can hurt deeply. Did something specific trigger it?"},
            ],
            "language_check": "en",
            "multi_turn_check": True,
        },
    ]

    seek_disallowed_terms = [
        "step 1", "step 2", "step 3", "step 4",
        "resilient zone", "body literacy", "practice check",
        "learning objective", "completed this objective", "skill_0",
    ]

    for scenario in test_scenarios:
        print(f"\n--- Scenario {scenario['id']}: {scenario['name']} ---")
        payload = {
            "message": scenario["message"],
            "history": scenario["history"],
            "request_id": f"e2e-scene-{scenario['id']}",
        }

        resp = client.post(
            "/api/general-empathy-chat",
            json=payload,
            headers=auth_headers,
        )

        status_code = resp.status_code
        data = resp.get_json() or {}
        response_text = data.get("response", "")

        has_seek_leak = any(term in response_text.lower() for term in seek_disallowed_terms)
        word_count = len(response_text.split())

        passed = status_code == 200 and bool(response_text) and not has_seek_leak

        if scenario.get("no_advice_check"):
            has_exercises = any(w in response_text.lower() for w in ["breathing exercise", "step 1", "first step"])
            if has_exercises:
                passed = False

        scenario_result = {
            "id": scenario["id"],
            "name": scenario["name"],
            "status_code": status_code,
            "response": response_text,
            "word_count": word_count,
            "has_seek_leak": has_seek_leak,
            "passed": passed,
        }
        results["conversations"].append(scenario_result)

        print(f"Status Code: {status_code}")
        print(f"Agent Response: {response_text}")
        print(f"Word Count: {word_count} | SEEK Leak: {has_seek_leak} | Result: {'PASS' if passed else 'FAIL'}")

    # -----------------------------------------------------------------------
    # 4. Safety Escalation & Lockout Verification
    # -----------------------------------------------------------------------
    log_section("4. SAFETY SCREENING & ESCALATION VERIFICATION")

    # A. Normal emotional expression does not trigger safety
    resp_safe = client.post(
        "/api/general-empathy-chat",
        json={"message": "I feel sad and tired after a hard week."},
        headers=auth_headers,
    )
    safe_data = resp_safe.get_json() or {}
    not_escalated = resp_safe.status_code == 200 and safe_data.get("safety") is not True
    print(f"Normal emotional sadness not escalated: {'PASS' if not_escalated else 'FAIL'}")
    results["safety"].append({"test": "Normal sadness not escalated", "passed": not_escalated})

    # B. Qualifying crisis message triggers safety response without LLM
    with patch.object(server.general_empathy_agent, "execute") as mock_agent_exec:
        resp_crisis = client.post(
            "/api/general-empathy-chat",
            json={"message": "I want to kill myself.", "request_id": "crisis-e2e-1"},
            headers=auth_headers,
        )
        crisis_data = resp_crisis.get_json() or {}
        crisis_triggered = (
            resp_crisis.status_code == 200
            and crisis_data.get("safety") is True
            and "response" not in crisis_data
            and crisis_data.get("counsellor_contacts") is not None
        )
        mock_agent_exec.assert_not_called()
        print(f"Red-flag crisis triggers immediate safety response (agent not called): {'PASS' if crisis_triggered else 'FAIL'}")
        results["safety"].append({"test": "Red-flag crisis escalation", "passed": crisis_triggered})

    # -----------------------------------------------------------------------
    # 5. Backend Security & Contract Verification
    # -----------------------------------------------------------------------
    log_section("5. BACKEND SECURITY, JWT ASSERTION & ERROR CONTRACTS")

    # A. Unauthenticated request rejected with 401
    resp_unauth = client.post(
        "/api/general-empathy-chat",
        json={"message": "Hello"},
    )
    unauth_pass = resp_unauth.status_code == 401
    print(f"Unauthenticated request returns 401: {'PASS' if unauth_pass else 'FAIL'}")
    results["backend_security"].append({"test": "401 on missing auth", "passed": unauth_pass})

    # B. Client-supplied student_id cannot override JWT
    with patch.object(server.general_empathy_agent, "execute") as mock_exec:
        mock_exec.return_value = GeneralEmpathyOutput(response="Test response")
        resp_tamper = client.post(
            "/api/general-empathy-chat",
            json={"message": "Hello", "student_id": "attacker_id"},
            headers=auth_headers,
        )
        assert resp_tamper.status_code == 200
        call_arg = mock_exec.call_args[0][0]
        actual_student_id = existing_student.get("student_id") or str(existing_student["_id"])
        tamper_pass = (
            call_arg.student_id == actual_student_id
            and call_arg.authenticated_student_id == actual_student_id
            and call_arg.student_id != "attacker_id"
        )
        print(f"Client-supplied student_id ignored in favor of JWT identity: {'PASS' if tamper_pass else 'FAIL'}")
        results["backend_security"].append({"test": "Tamper-proof JWT identity", "passed": tamper_pass})

    # C. Gemini outage returns HTTP 503
    with patch.object(server.general_empathy_agent, "execute", side_effect=server.AgentExecutionError("Down")):
        resp_503 = client.post(
            "/api/general-empathy-chat",
            json={"message": "Hello"},
            headers=auth_headers,
        )
        err_503_pass = resp_503.status_code == 503 and "unavailable" in (resp_503.get_json() or {}).get("error", "")
        print(f"Agent failure returns HTTP 503: {'PASS' if err_503_pass else 'FAIL'}")
        results["backend_security"].append({"test": "HTTP 503 on agent failure", "passed": err_503_pass})

    # -----------------------------------------------------------------------
    # 6. Database Immutability & SEEK Preservation
    # -----------------------------------------------------------------------
    log_section("6. DATABASE IMMUTABILITY & SEEK PRESERVATION")

    if db is not None:
        try:
            post_collections = sorted(db.list_collection_names())
            post_progress_count = db["student_progress"].count_documents({})
            post_interactions_count = db["learning_interactions"].count_documents({})

            collections_unchanged = (pre_collections == post_collections)
            progress_unchanged = (pre_progress_count == post_progress_count)
            interactions_unchanged = (pre_interactions_count == post_interactions_count)

            print(f"Collections unchanged: {collections_unchanged} ({post_collections})")
            print(f"student_progress unchanged: {progress_unchanged} (count={post_progress_count})")
            print(f"learning_interactions unchanged: {interactions_unchanged} (count={post_interactions_count})")

            results["database_integrity"] = {
                "collections_unchanged": collections_unchanged,
                "progress_unchanged": progress_unchanged,
                "interactions_unchanged": interactions_unchanged,
            }
        except Exception as e:
            print(f"Database check exception: {e}")

    # Verify SEEK learning pathway still generates 4-stage response
    resp_seek = client.post(
        "/api/learning-response",
        json={"question": "What is the resilient zone?", "answer_style": "short"},
        headers=auth_headers,
    )
    seek_data = resp_seek.get_json() or {}
    seek_works = (
        resp_seek.status_code == 200
        and "educational_response" in seek_data
        and len(seek_data.get("steps", [])) == 4
    )
    print(f"SEEK Learning pathway generates 4-stage educational contract: {'PASS' if seek_works else 'FAIL'}")
    results["seek_preservation"] = {"passed": seek_works}

    return results


if __name__ == "__main__":
    verification_results = run_e2e_verification()
    output_path = ROOT / "phase5_e2e_verification_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    print(f"\nVerification finished. Report saved to: {output_path}")
