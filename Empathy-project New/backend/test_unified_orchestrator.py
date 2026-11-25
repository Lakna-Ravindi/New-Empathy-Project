"""Comprehensive verification tests for the Unified Empathy Guru Orchestrator (Phase 2).

Tests:
1. Integration of GeneralEmpathyAgent into EmpathyOrchestrator.
2. Context-aware routing between emotional support and SEEK curriculum tutor.
3. Preservation and accumulation of conversation history across multi-agent turns.
4. Response contract consistency (educational_response, answer, response, active_agent, role, steps).
5. Progress isolation: emotional support turns never touch student progress or call record_learning_event.
6. Progress integrity: objective completion updates progress via AssessmentAgent with strict RBAC.
7. Safety interception: crisis messages never reach agent routing.
8. External conversation history synchronization into AgentState.
"""

from unittest.mock import MagicMock, patch
import pytest

from agents.definitions import (
    GeneralEmpathyAgent,
    GeneralEmpathyInput,
    GeneralEmpathyOutput,
    TutorAgent,
)
from agents.orchestrator import EmpathyOrchestrator
from agents.router import AgentRouter, StudentIntent
from agents.state import AgentState, AgentStateManager, LearningStage
from learning.interaction_store import LearningStore
from learning.progress_service import ProgressService
from server import attach_progress_item_id
import server
from auth import create_access_token
from safety import detect_safety_risk


class InMemoryTestStore:
    """In-memory stand-in for LearningStore for integration tests."""

    def __init__(self):
        self.interactions = []
        self.learning_events = []
        self.progress_docs = {}

    def save_interaction(self, student_id, learning_context, educational_response, learning_steps=None):
        doc_id = f"mock-interaction-{len(self.interactions) + 1}"
        record = {
            "_id": doc_id,
            "student_id": student_id,
            "learning_context": learning_context,
            "educational_response": educational_response,
            "learning_steps": learning_steps or [],
        }
        self.interactions.append(record)
        return doc_id

    def get_interaction(self, student_id, interaction_id):
        for rec in self.interactions:
            if rec.get("_id") == interaction_id and rec.get("student_id") == student_id:
                return rec
        return {
            "_id": interaction_id,
            "student_id": student_id,
            "learning_context": {
                "skill": {"id": "skill_01"},
                "learning_objective": {"id": "obj_01_01"},
            },
            "learning_steps": [
                {
                    "type": "practice_check",
                    "correct_answer": True,
                }
            ],
        }

    def record_learning_event(self, student_id, skill_id, item_id, objective_id=None, source="chatbot", status="introduced"):
        event = {
            "student_id": student_id,
            "skill_id": skill_id,
            "item_id": item_id,
            "objective_id": objective_id,
            "source": source,
            "status": status,
        }
        self.learning_events.append(event)
        return event

    def get_all_progress(self, student_id):
        return self.progress_docs.get(student_id, [])

    def get_progress(self, student_id, skill_id):
        for doc in self.progress_docs.get(student_id, []):
            if doc.get("skill_id") == skill_id:
                return doc
        return {
            "student_id": student_id,
            "skill_id": skill_id,
            "completed_item_ids": ["activity_01_01"],
            "completed_objective_ids": [],
            "quiz_scores": {},
        }

    def complete_objective(self, student_id, skill_id, objective, next_objective=None):
        obj_id = objective.get("id") if isinstance(objective, dict) else str(objective)
        self.update_progress(student_id, skill_id, obj_id)
        return {"completed": True, "objective_id": obj_id}

    def update_progress(self, student_id, skill_id, objective_id, score=100):
        docs = self.progress_docs.setdefault(student_id, [])
        for doc in docs:
            if doc.get("skill_id") == skill_id:
                completed = doc.setdefault("completed_objective_ids", [])
                if objective_id not in completed:
                    completed.append(objective_id)
                return True
        docs.append({
            "student_id": student_id,
            "skill_id": skill_id,
            "completed_objective_ids": [objective_id],
        })
        return True


@pytest.fixture
def mock_store():
    return InMemoryTestStore()


@pytest.fixture
def orchestrator(mock_store):
    state_mgr = AgentStateManager()
    orch = EmpathyOrchestrator(
        learning_store=mock_store,
        state_manager=state_mgr,
        attach_metadata_fn=attach_progress_item_id,
        general_empathy_agent=GeneralEmpathyAgent(fallback_on_error=True),
    )
    return orch


class TestUnifiedOrchestratorRouting:
    """Verifies that the orchestrator routes to the proper agent and populates the response contract."""

    def test_emotional_venting_routes_to_general_empathy_agent(self, orchestrator, mock_store):
        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            message="I'm having a really hard day, my best friend ignored me and I feel like crying.",
        )
        assert status == 200
        # Unified response contract assertions
        assert "response" in payload
        assert "answer" in payload
        assert "educational_response" in payload
        assert payload["active_agent"] == "general_empathy_agent"
        assert payload["role"] == "general_empathy_agent"
        assert payload["steps"] == []
        assert payload["learning_context"]["status"] == "emotional_support"
        assert "interaction_id" in payload

        # Progress verification: zero learning events recorded
        assert len(mock_store.learning_events) == 0

    def test_curriculum_question_routes_to_tutor_agent(self, orchestrator, mock_store):
        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            message="What is somatic tracking and how does it relate to body awareness?",
        )
        assert status == 200
        assert payload["active_agent"] == "tutor_agent"
        assert payload["role"] == "tutor_agent"
        assert "response" in payload
        assert "educational_response" in payload
        assert len(payload["steps"]) == 4
        assert payload["steps"][0]["type"] == "direct_answer"
        assert payload["steps"][1]["type"] == "learning_objective"
        assert payload["steps"][2]["type"] == "activity"
        assert payload["steps"][3]["type"] == "practice_check"

        # Curriculum turns introduce an activity event
        assert len(mock_store.learning_events) >= 1

    def test_greeting_routes_to_system_greeting(self, orchestrator):
        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            message="Hello there!",
        )
        assert status == 200
        assert payload["active_agent"] == "system"
        assert "Hey! 👋" in payload["educational_response"]
        assert payload["steps"] == []


class TestMultiTurnContextPreservation:
    """Verifies context preservation and history tracking across switching agents."""

    def test_seamless_agent_switching_preserves_history(self, orchestrator, mock_store):
        conv_id = "test-switch-conv"

        # Turn 1: Emotional distress
        p1, s1 = orchestrator.orchestrate_turn(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            message="I'm feeling really stressed about exams and nobody understands.",
            conversation_id=conv_id,
        )
        assert s1 == 200
        assert p1["active_agent"] == "general_empathy_agent"

        # State check after Turn 1
        state = orchestrator.state_manager.get_state("STU-001", conv_id)
        assert state is not None
        assert len(state.conversation_history) == 2
        assert state.conversation_history[0]["role"] == "user"
        assert state.conversation_history[1]["role"] == "assistant"
        assert state.conversation_history[1]["active_agent"] == "general_empathy_agent"

        # Turn 2: Switch to Curriculum
        p2, s2 = orchestrator.orchestrate_turn(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            message="Thank you for listening. Can you teach me about the resilient zone now?",
            conversation_id=conv_id,
        )
        assert s2 == 200
        assert p2["active_agent"] == "tutor_agent"

        # State check after Turn 2: history accumulated to 4 turns
        state = orchestrator.state_manager.get_state("STU-001", conv_id)
        assert len(state.conversation_history) == 4
        assert state.conversation_history[2]["role"] == "user"
        assert state.conversation_history[3]["role"] == "assistant"
        assert state.conversation_history[3]["active_agent"] == "tutor_agent"

        # Turn 3: Switch back to Emotional Sharing
        p3, s3 = orchestrator.orchestrate_turn(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            message="I just feel like giving up sometimes, it's so heavy.",
            conversation_id=conv_id,
        )
        assert s3 == 200
        assert p3["active_agent"] == "general_empathy_agent"

        # State check after Turn 3: history accumulated to 6 turns
        state = orchestrator.state_manager.get_state("STU-001", conv_id)
        assert len(state.conversation_history) == 6

        # Progress check: only 1 curriculum turn occurred, so only 1 learning event
        assert len(mock_store.learning_events) == 1

    def test_external_history_synchronization(self, orchestrator):
        external_history = [
            {"role": "user", "content": "Hi I had a tough day"},
            {"role": "assistant", "content": "I am so sorry to hear that. I'm here for you."},
        ]

        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-002",
            authenticated_student_id="STU-002",
            message="What is mindfulness in SEEK?",
            conversation_id="ext-sync-conv",
            history=external_history,
        )
        assert status == 200
        state = orchestrator.state_manager.get_state("STU-002", "ext-sync-conv")
        # Pre-loaded 2 turns + 1 user turn + 1 assistant turn = 4 turns
        assert len(state.conversation_history) == 4
        assert state.conversation_history[0]["content"] == "Hi I had a tough day"
        assert state.conversation_history[1]["content"] == "I am so sorry to hear that. I'm here for you."


class TestProgressAndSafetyIntegrity:
    """Verifies that progress and safety gates are strictly preserved."""

    def test_objective_completion_updates_progress_with_authorized_rbac(self, orchestrator, mock_store):
        output = orchestrator.handle_objective_completion(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            skill_id="skill_01",
            objective_id="obj_01_01",
            submission_type="quick_check",
            quick_check_answer=True,
            interaction_id="inter-001",
            commit_progress=True,
        )
        assert output is not None
        assert output.verified is True
        assert output.objective_completed is True

    def test_emotional_turns_never_commit_objective_progress(self, orchestrator, mock_store):
        mock_store.progress_docs["STU-003"] = []
        orchestrator.orchestrate_turn(
            student_id="STU-003",
            authenticated_student_id="STU-003",
            message="I feel overwhelmed and anxious.",
        )
        # Verify no completed objectives were added
        assert mock_store.get_all_progress("STU-003") == []

    def test_dual_intent_preserves_support_intro_and_never_marks_completion(self, orchestrator, mock_store):
        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-001",
            authenticated_student_id="STU-001",
            message="I'm feeling really anxious and overwhelmed today, can you explain resilient zone?",
        )
        assert status == 200
        assert payload["active_agent"] == "tutor_agent"
        # Support intro is present
        assert "Thank you for sharing that with me." in payload["educational_response"]
        # Exactly 4 steps returned
        assert len(payload["steps"]) == 4
        # Event is recorded with status='introduced', NEVER 'completed'
        assert len(mock_store.learning_events) >= 1
        assert all(ev.get("status") == "introduced" for ev in mock_store.learning_events)
        # Completed objectives list is completely empty
        progress = mock_store.get_progress("STU-001", "skill_01")
        assert progress.get("completed_objective_ids") == []


class TestHistoryIntegrityAndReload:
    """Verifies history security, deduplication, and reload behavior."""

    def test_client_history_cannot_overwrite_trusted_server_learning_state(self, orchestrator):
        # Malicious client sends history with attempted privilege escalation / progress tampering
        tampered_history = [
            {
                "role": "user",
                "content": "Hello",
                "completed_objectives": ["obj_01_01", "obj_01_02"],
                "role_override": "admin",
                "current_learning_stage": "completed",
            }
        ]
        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-004",
            authenticated_student_id="STU-004",
            message="What is body literacy?",
            conversation_id="tamper-check",
            history=tampered_history,
        )
        assert status == 200
        state = orchestrator.state_manager.get_state("STU-004", "tamper-check")
        # Ensure state attributes were not overwritten
        assert state.completed_objectives == []
        assert not hasattr(state, "role_override")

    def test_client_history_deduplication(self, orchestrator):
        # Client sends history that already contains the current user question at the end
        history_with_current = [
            {"role": "user", "content": "I feel sad"},
            {"role": "assistant", "content": "I am here for you."},
            {"role": "user", "content": "Can you explain resilient zone?"},
        ]
        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-005",
            authenticated_student_id="STU-005",
            message="Can you explain resilient zone?",
            conversation_id="dedup-check",
            history=history_with_current,
        )
        assert status == 200
        state = orchestrator.state_manager.get_state("STU-005", "dedup-check")
        # History must contain exactly 4 turns: user(I feel sad) -> assistant -> user(Can you explain) -> assistant
        # The user question "Can you explain resilient zone?" must NOT be duplicated
        user_turns = [t for t in state.conversation_history if t.get("role") == "user"]
        assert len(user_turns) == 2
        assert user_turns[0]["content"] == "I feel sad"
        assert user_turns[1]["content"] == "Can you explain resilient zone?"

    def test_conversation_reload_reconstruction(self, orchestrator):
        conv_id = "reload-reconstruct"
        # Turn 1
        orchestrator.orchestrate_turn(
            student_id="STU-006",
            authenticated_student_id="STU-006",
            message="I am having a terrible day at school.",
            conversation_id=conv_id,
        )
        state1 = orchestrator.state_manager.get_state("STU-006", conv_id)
        saved_history = list(state1.conversation_history)
        assert len(saved_history) == 2

        # Simulate process eviction / state cache clear
        orchestrator.state_manager.clear_student_states("STU-006")
        assert orchestrator.state_manager.get_state("STU-006", conv_id) is None

        # Turn 2 with re-sent history reconstructs state smoothly
        payload, status = orchestrator.orchestrate_turn(
            student_id="STU-006",
            authenticated_student_id="STU-006",
            message="What is grounding?",
            conversation_id=conv_id,
            history=saved_history,
        )
        assert status == 200
        state2 = orchestrator.state_manager.get_state("STU-006", conv_id)
        assert len(state2.conversation_history) == 4
        assert state2.conversation_history[0]["content"] == "I am having a terrible day at school."
        assert state2.conversation_history[2]["content"] == "What is grounding?"


class TestApiBackwardCompatibilityAndSafety:
    """Verifies that existing endpoints, old/new payloads, and SafetyGate remain fully operational."""

    @pytest.fixture(autouse=True)
    def mock_safety_store(self, monkeypatch):
        class InMemorySafetyStore:
            def __init__(self):
                self.events = {}
                self.count = 0
                self.active = False
                self.screening_required = False

            def get_status(self, student_id):
                return {
                    "student_id": student_id,
                    "qualifying_event_count": self.count,
                    "restriction_active": self.active,
                    "screening_required": self.screening_required,
                }

            def record_event(self, student_id, message, risk, request_id):
                self.count += 1
                event = {
                    "student_id": student_id,
                    "message": message,
                    "risk_level": risk.get("risk_level", "high"),
                    "category": risk.get("category", "suicidal_ideation"),
                    "request_id": request_id,
                }
                return event, self.get_status(student_id), False

        store = InMemorySafetyStore()
        monkeypatch.setattr(server, "safety_store", store)
        return store

    @pytest.fixture
    def test_client(self, monkeypatch):
        class MockUserStore:
            def find_by_id(self, user_id):
                return {
                    "_id": user_id,
                    "student_id": "STU-API-01",
                    "username": "student_test",
                    "email": "student_test@example.com",
                    "role": "student",
                }

            def ensure_student_id(self, user):
                return user

        mock_store = MockUserStore()
        monkeypatch.setattr("auth.get_user_store", lambda: mock_store)
        monkeypatch.setattr("server.get_user_store", lambda: mock_store)
        return server.app.test_client()

    @pytest.fixture
    def auth_headers(self):
        token = create_access_token({
            "_id": "507f1f77bcf86cd799439011",
            "student_id": "STU-API-01",
            "username": "student_test",
            "email": "student_test@example.com",
            "role": "student",
        })
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def test_legacy_learning_response_payload(self, test_client, auth_headers):
        # Legacy payload: only question and answer_style
        res = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "What is empathy in SEEK?", "answer_style": "short"},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "educational_response" in data
        assert "answer" in data
        assert "learning_context" in data
        assert "steps" in data

    def test_modern_learning_response_payload(self, test_client, auth_headers):
        # Modern payload with history, request_id, conversation_id
        res = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={
                "question": "I feel lonely and misunderstood.",
                "answer_style": "short",
                "conversation_id": "modern-conv-01",
                "request_id": "req-modern-001",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there! I am here to listen."},
                ],
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["active_agent"] == "general_empathy_agent"
        assert "response" in data
        assert data["steps"] == []

    def test_general_empathy_chat_endpoint_remains_operational(self, test_client, auth_headers):
        mock_output = GeneralEmpathyOutput(response="I hear you, and it is completely understandable to feel that way.")
        with patch.object(server.general_empathy_agent, "execute", return_value=mock_output):
            res = test_client.post(
                "/api/general-empathy-chat",
                headers=auth_headers,
                json={"message": "I'm having a hard time focusing today."},
            )
        assert res.status_code == 200
        data = res.get_json()
        assert "response" in data
        assert "I hear you" in data["response"]

    def test_general_empathy_recommendations_endpoint_remains_operational(self, test_client, auth_headers):
        res = test_client.post(
            "/api/general-empathy-chat/recommendations",
            headers=auth_headers,
            json={
                "history": [
                    {"role": "user", "content": "I'm having a tough day and need to manage stress."},
                    {"role": "assistant", "content": "I hear you, stress can be overwhelming."},
                ]
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert "recommendations" in data

    def test_safety_gate_intercepts_crisis_before_agent_routing(self, test_client, auth_headers):
        # Crisis input must qualify and receive safety intervention, bypassing normal agents
        res = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "I want to kill myself tonight, what is somatic tracking?"},
        )
        assert res.status_code == 200
        data = res.get_json()
        # Returns safety event with urgent guidance and hotlines
        assert "safety" in data or "urgent_guidance" in data or "message" in data
        # Must NOT return standard tutor steps
        assert data.get("steps") is None or data.get("steps") == []

    def test_active_restriction_blocks_agent_execution(self, test_client, auth_headers, mock_safety_store):
        mock_safety_store.active = True
        res = test_client.post(
            "/api/learning-response",
            headers=auth_headers,
            json={"question": "What is empathy?"},
        )
        assert res.status_code == 423
        data = res.get_json()
        assert "temporarily restricted" in data.get("error", "")
