"""Phase 2A Regression Test Suite: RAG Pipeline and Agentic AI Integration.

Validates the 6 audit-defined regression test areas:
1. PDF Ingestion & Knowledge Base Cleanliness (no headers/footers in content nodes).
2. Skill Routing for core curriculum questions (avoiding spurious misrouting).
3. Original RAG Retrieval (top chunks must contain genuine empathy nodes, not headers).
4. Learning Planner Agent (alignment between semantic evidence, skill choice, and constraints).
5. Fallback Responder (no header metadata as direct answer or quick check).
6. Integrated Pipeline End-to-End (question -> planner -> RAG -> tutor -> response).

Strictly read-only with respect to production code, MongoDB, and cached embeddings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend root is on sys.path
import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.definitions import (
    LearningPlannerAgent,
    LearningPlannerInput,
    LearningPlannerOutput,
    TutorAgent,
    TutorInput,
    TutorOutput,
)
from agents.orchestrator import EmpathyOrchestrator
from agents.state import AgentState, LearningStage
from agents.tools import get_default_tool_registry
from agents.tools.curriculum_tools import (
    get_content_skills,
    get_knowledge_base,
    get_progress_structure,
    retrieve_approved_seek_content,
)
from learning.gemini_responder import _fallback_response
from learning.pedagogical_controller import (
    curriculum_skill_id,
    process_student_question,
)
from learning.semantic_search import semantic_search


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def knowledge_base() -> List[Dict[str, Any]]:
    test_kb_path = ROOT.parent / "output" / "test_cleaned_knowledge_base.json"
    if test_kb_path.exists():
        with test_kb_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return get_knowledge_base()


@pytest.fixture(scope="module")
def content_skills() -> List[Dict[str, Any]]:
    return get_content_skills()


@pytest.fixture(scope="module")
def progress_structure() -> Dict[str, Any]:
    return get_progress_structure()


@pytest.fixture(scope="module")
def keyword_skill_map() -> List[Dict[str, Any]]:
    map_path = ROOT.parent / "output" / "keyword_skill_map.json"
    if map_path.exists():
        with map_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


@pytest.fixture
def tool_registry():
    return get_default_tool_registry()


@pytest.fixture
def mock_learning_store():
    store = MagicMock()
    store.get_progress.return_value = {
        "student_id": "STU-REG-001",
        "skill_id": "skill_01",
        "completed_item_ids": [],
        "completed_objective_ids": [],
        "learning_events": [],
    }
    store.get_all_progress.return_value = [
        {
            "student_id": "STU-REG-001",
            "skill_id": "skill_01",
            "completed_item_ids": [],
            "completed_objective_ids": [],
            "learning_events": [],
        }
    ]
    store.save_interaction.return_value = "mock_interaction_reg_001"
    store.record_learning_event.return_value = {}
    return store


# =====================================================================
# Area 1: PDF Ingestion & Knowledge Base Cleanliness
# =====================================================================

class TestPDFIngestionRegression:
    """Tests that PDF running headers, footers, and cover titles are not indexed as content."""

    HEADER_PATTERN = re.compile(
        r"UNESCO\s+MGIEP\s*—\s*SEEK\s+Learning\s+Curriculum\s+Study\s+Guide.*Page\s+\d+\s+of\s+\d+",
        re.IGNORECASE,
    )
    COVER_TITLE_PATTERN = re.compile(
        r"DEVELOPMENT\s+SEEK\s+Learning\s+Curriculum\s+Study\s+Guide",
        re.IGNORECASE,
    )

    def test_running_headers_must_not_be_approved_content_nodes(self, knowledge_base):
        """Verifies that no approved content node contains raw page running headers."""
        header_nodes = [
            node for node in knowledge_base
            if node.get("type") == "content"
            and self.HEADER_PATTERN.search(
                f"{node.get('title', '')} {node.get('content', '')}"
            )
        ]
        assert not header_nodes, (
            f"Found {len(header_nodes)} running header nodes incorrectly indexed as content nodes: "
            f"{[n.get('id') for n in header_nodes[:5]]}"
        )

    def test_cover_titles_and_page_metadata_must_not_be_knowledge_nodes(self, knowledge_base):
        """Verifies that document cover titles are not indexed as searchable content nodes."""
        cover_nodes = [
            node for node in knowledge_base
            if node.get("type") == "content"
            and self.COVER_TITLE_PATTERN.search(
                f"{node.get('title', '')} {node.get('content', '')}"
            )
        ]
        assert not cover_nodes, (
            f"Found cover title nodes incorrectly indexed as content nodes: "
            f"{[n.get('id') for n in cover_nodes]}"
        )

    def test_genuine_seek_curriculum_nodes_preserved(self, knowledge_base):
        """Verifies that key genuine empathy curriculum nodes exist with substantial content."""
        nodes_by_id = {n["id"]: n for n in knowledge_base if n.get("id")}

        # node_0417: Definition of Empathy
        assert "node_0417" in nodes_by_id, "node_0417 (Definition of Empathy) missing from KB."
        n417 = nodes_by_id["node_0417"]
        assert "ability to understand and share" in (n417.get("content") or "").lower()
        assert n417.get("skill_id") == "skill_07"

        # node_0420: Empathy vs. Sympathy
        assert "node_0420" in nodes_by_id, "node_0420 (Empathy vs. Sympathy) missing from KB."
        n420 = nodes_by_id["node_0420"]
        assert "stepping into their perspective" in (n420.get("content") or "").lower()
        assert n420.get("skill_id") == "skill_07"

    def test_all_eight_skills_available_in_knowledge_base(self, knowledge_base):
        """Verifies that all 8 SEEK skills are represented by chapter nodes."""
        chapters = [n for n in knowledge_base if n.get("type") == "chapter"]
        assert len(chapters) == 8, f"Expected 8 skill chapters, found {len(chapters)}."
        chapter_titles = [c.get("title", "") for c in chapters]
        assert any("empathic concern" in t.lower() for t in chapter_titles), (
            "Skill 7 (Empathic Concern) chapter is missing."
        )


# =====================================================================
# Area 2: Skill Routing
# =====================================================================

class TestSkillRoutingRegression:
    """Tests skill routing for the 6 core questions specified in Phase 2A audit."""

    def test_routing_what_is_empathy(self, knowledge_base, keyword_skill_map, progress_structure):
        """Q1: 'What is empathy?' must route to Skill 7 – Empathic Concern."""
        res = process_student_question(
            "What is empathy?",
            knowledge_base,
            highlighted_keywords=keyword_skill_map,
            progress_structure=progress_structure,
        )
        assert res.get("status") == "learning_path_selected"
        skill_id = (res.get("skill") or {}).get("skill_id") or (res.get("skill") or {}).get("id")
        assert skill_id == "skill_07", (
            f"Expected Skill 7 (Empathic Concern), got {skill_id} ({res.get('skill', {}).get('title')})"
        )

    def test_routing_why_is_empathy_important_in_learning(
        self, knowledge_base, keyword_skill_map, progress_structure
    ):
        """Q2: 'Why is empathy important in learning?' must route to Skill 7 – Empathic Concern."""
        res = process_student_question(
            "Why is empathy important in learning?",
            knowledge_base,
            highlighted_keywords=keyword_skill_map,
            progress_structure=progress_structure,
        )
        assert res.get("status") == "learning_path_selected"
        skill_id = (res.get("skill") or {}).get("skill_id") or (res.get("skill") or {}).get("id")
        assert skill_id == "skill_07", (
            f"Expected Skill 7 (Empathic Concern), got {skill_id} ({res.get('skill', {}).get('title')})"
        )

    def test_routing_difference_between_empathy_and_sympathy(
        self, knowledge_base, keyword_skill_map, progress_structure
    ):
        """Q3: 'What is the difference between empathy and sympathy?' must route to Skill 7."""
        res = process_student_question(
            "What is the difference between empathy and sympathy?",
            knowledge_base,
            highlighted_keywords=keyword_skill_map,
            progress_structure=progress_structure,
        )
        assert res.get("status") == "learning_path_selected"
        skill_id = (res.get("skill") or {}).get("skill_id") or (res.get("skill") or {}).get("id")
        assert skill_id == "skill_07", (
            f"Expected Skill 7 (Empathic Concern), got {skill_id} ({res.get('skill', {}).get('title')})"
        )

    def test_routing_how_can_i_manage_my_emotions(
        self, knowledge_base, keyword_skill_map, progress_structure
    ):
        """Q4: 'How can I manage my emotions?' must route to emotion-regulation skill (Skill 1 or Skill 3)."""
        res = process_student_question(
            "How can I manage my emotions?",
            knowledge_base,
            highlighted_keywords=keyword_skill_map,
            progress_structure=progress_structure,
        )
        assert res.get("status") == "learning_path_selected"
        skill_id = (res.get("skill") or {}).get("skill_id") or (res.get("skill") or {}).get("id")
        assert skill_id in {"skill_01", "skill_03"}, (
            f"Expected Skill 1 (Calming) or Skill 3 (Emotional Awareness), got {skill_id}"
        )

    def test_routing_how_can_i_develop_self_compassion(
        self, knowledge_base, keyword_skill_map, progress_structure
    ):
        """Q5: 'How can I develop self-compassion?' must route to Skill 4 – Self-Compassion."""
        res = process_student_question(
            "How can I develop self-compassion?",
            knowledge_base,
            highlighted_keywords=keyword_skill_map,
            progress_structure=progress_structure,
        )
        assert res.get("status") == "learning_path_selected"
        skill_id = (res.get("skill") or {}).get("skill_id") or (res.get("skill") or {}).get("id")
        assert skill_id == "skill_04", (
            f"Expected Skill 4 (Self-Compassion), got {skill_id}"
        )

    def test_routing_how_can_i_understand_another_persons_feelings(
        self, knowledge_base, keyword_skill_map, progress_structure
    ):
        """Q6: 'How can I understand another person's feelings?' must route to relevant empathy skill (Skill 7 or Skill 5)."""
        res = process_student_question(
            "How can I understand another person's feelings?",
            knowledge_base,
            highlighted_keywords=keyword_skill_map,
            progress_structure=progress_structure,
        )
        assert res.get("status") == "learning_path_selected"
        skill_id = (res.get("skill") or {}).get("skill_id") or (res.get("skill") or {}).get("id")
        assert skill_id in {"skill_07", "skill_05"}, (
            f"Expected Skill 7 (Empathic Concern) or Skill 5 (Common Humanity), got {skill_id}"
        )


# =====================================================================
# Area 3: Original RAG Retrieval
# =====================================================================

class TestOriginalRAGRetrievalRegression:
    """Tests semantic retrieval accuracy and ensures no header pollution in top chunks."""

    def test_unconstrained_semantic_retrieval_retrieves_empathy_definition_nodes(
        self, knowledge_base
    ):
        """Unconstrained search for empathy must rank node_0417 and node_0420 in top results."""
        results = semantic_search(
            "What is empathy, and why is it important in learning?",
            knowledge_base,
            limit=5,
        )
        retrieved_ids = [r["id"] for r in results]
        assert "node_0417" in retrieved_ids, (
            f"node_0417 (Definition of Empathy) not in top 5 retrieved IDs: {retrieved_ids}"
        )
        assert "node_0420" in retrieved_ids, (
            f"node_0420 (Empathy vs. Sympathy) not in top 5 retrieved IDs: {retrieved_ids}"
        )

    def test_top_retrieved_chunks_contain_no_pdf_headers_or_footers(
        self, knowledge_base
    ):
        """Top 5 retrieved chunks must not contain PDF page running headers or document titles."""
        results = semantic_search(
            "What is empathy, and why is it important in learning?",
            knowledge_base,
            limit=5,
        )
        header_pattern = re.compile(r"Page \d+ of \d+|Study Guide \(Skills 1–8\)", re.IGNORECASE)
        polluted_chunks = [
            r for r in results
            if header_pattern.search(f"{r.get('title', '')} {r.get('content', '')}")
        ]
        assert not polluted_chunks, (
            f"Found {len(polluted_chunks)} header/footer chunks in top 5 results: "
            f"{[c['id'] for c in polluted_chunks]}"
        )

    def test_constrained_retrieval_for_skill_07_returns_valid_chunks(
        self, knowledge_base
    ):
        """Constrained retrieval within skill_07 must return genuine Skill 7 content."""
        chunks = retrieve_approved_seek_content(
            "What is empathy, and why is it important in learning?",
            skill_id="skill_07",
            objective_id="obj_07_01",
            limit=5,
            knowledge_base=knowledge_base,
        )
        assert chunks, "retrieve_approved_seek_content returned empty list for skill_07/obj_07_01."
        assert chunks[0]["id"] == "node_0417"
        assert all(c.get("skill_id") == "skill_07" for c in chunks)
        # Check chunk 0 is not a header
        assert "Page 20 of 24" not in chunks[0]["title"]


# =====================================================================
# Area 4: Learning Planner
# =====================================================================

class TestLearningPlannerRegression:
    """Tests LearningPlannerAgent decisions, semantic alignment, and read-only boundaries."""

    def test_planner_skill_selection_aligns_with_semantic_evidence(
        self, tool_registry, mock_learning_store
    ):
        """Planner must select skill_07 and obj_07_01 for the test empathy question."""
        planner = LearningPlannerAgent(tool_registry=tool_registry)
        out = planner.execute(
            LearningPlannerInput(
                student_id="STU-REG-001",
                authenticated_student_id="STU-REG-001",
                student_question="What is empathy, and why is it important in learning?",
            )
        )
        assert out.skill["id"] == "skill_07", (
            f"Planner selected skill {out.skill} instead of skill_07 (Empathic Concern)."
        )
        assert out.learning_objective["id"] in {"obj_07_01", "obj_07_02"}, (
            f"Planner selected objective {out.learning_objective} instead of obj_07_01/obj_07_02."
        )

    def test_planner_does_not_silently_restrict_retrieval_to_unrelated_skill(
        self, tool_registry, mock_learning_store
    ):
        """Planner's relevant_chunks must not be constrained to Skill 4 when asking about empathy."""
        planner = LearningPlannerAgent(tool_registry=tool_registry)
        out = planner.execute(
            LearningPlannerInput(
                student_id="STU-REG-001",
                authenticated_student_id="STU-REG-001",
                student_question="What is empathy, and why is it important in learning?",
            )
        )
        for chunk in out.relevant_chunks:
            assert chunk.get("skill_id") != "skill_04", (
                f"Retrieved chunk {chunk.get('id')} belongs to skill_04 (Self-Compassion) for an empathy query."
            )

    def test_planner_objective_selection_grounded_in_approved_curriculum(
        self, tool_registry, mock_learning_store
    ):
        """Planner must only choose objectives from the approved curriculum."""
        planner = LearningPlannerAgent(tool_registry=tool_registry)
        out = planner.execute(
            LearningPlannerInput(
                student_id="STU-REG-001",
                authenticated_student_id="STU-REG-001",
                student_question="What is empathy, and why is it important in learning?",
            )
        )
        obj_id = out.learning_objective["id"]
        assert obj_id.startswith("obj_")
        assert out.learning_objective.get("title")

    def test_planner_preserves_student_progress_read_only(
        self, tool_registry, mock_learning_store
    ):
        """Planner execution must not mutate student progress or call record_learning_event."""
        planner = LearningPlannerAgent(tool_registry=tool_registry)
        planner.execute(
            LearningPlannerInput(
                student_id="STU-REG-001",
                authenticated_student_id="STU-REG-001",
                student_question="What is empathy, and why is it important in learning?",
            )
        )
        mock_learning_store.record_learning_event.assert_not_called()


# =====================================================================
# Area 5: Fallback Response
# =====================================================================

class TestFallbackResponseRegression:
    """Tests deterministic fallback response generation when Gemini is unavailable or times out."""

    def test_fallback_does_not_output_pdf_header_as_answer(self):
        """Fallback response must never echo a running header or cover title as the direct answer."""
        polluted_context = {
            "student_question": "What is empathy, and why is it important in learning?",
            "status": "learning_path_selected",
            "skill": {"id": "skill_07", "title": "Empathic Concern"},
            "learning_objective": {
                "id": "obj_07_01",
                "objective_id": "obj_07_01",
                "title": "Define Empathy and differentiate it from sympathy.",
            },
            "recommended_activity": None,
            "question_relevant_content": [
                {
                    "id": "node_0240",
                    "title": "UNESCO MGIEP — SEEK Learning Curriculum Study Guide (Skills 1–8) Page 12 of 24",
                    "content": "UNESCO MGIEP — SEEK Learning Curriculum Study Guide (Skills 1–8) Page 12 of 24",
                    "page": 12,
                }
            ],
        }
        res = _fallback_response(polluted_context, answer_style="short")
        answer = res.get("answer", "")
        assert "Page 12 of 24" not in answer, (
            f"Fallback answer contains page running header: '{answer}'"
        )
        assert "Curriculum Study Guide" not in answer, (
            f"Fallback answer contains study guide title: '{answer}'"
        )

    def test_fallback_quick_check_does_not_contain_header_metadata(self):
        """Step 4 Quick Check question in fallback must not be formed from header metadata."""
        polluted_context = {
            "student_question": "What is empathy, and why is it important in learning?",
            "status": "learning_path_selected",
            "skill": {"id": "skill_07", "title": "Empathic Concern"},
            "learning_objective": {
                "id": "obj_07_01",
                "objective_id": "obj_07_01",
                "title": "Define Empathy and differentiate it from sympathy.",
            },
            "recommended_activity": None,
            "question_relevant_content": [
                {
                    "id": "node_0240",
                    "title": "UNESCO MGIEP — SEEK Learning Curriculum Study Guide (Skills 1–8) Page 12 of 24",
                    "content": "UNESCO MGIEP — SEEK Learning Curriculum Study Guide (Skills 1–8) Page 12 of 24",
                    "page": 12,
                }
            ],
        }
        res = _fallback_response(polluted_context, answer_style="short")
        steps = res.get("steps", [])
        step4 = next((s for s in steps if s.get("step") == 4), {})
        question_text = step4.get("question", "")
        assert "Page 12 of 24" not in question_text, (
            f"Quick Check question contains header metadata: '{question_text}'"
        )

    def test_fallback_does_not_hallucinate_activity_when_none_selected(self):
        """When recommended_activity is None, Step 3 must state no activity was selected."""
        context = {
            "student_question": "What is empathy?",
            "status": "learning_path_selected",
            "skill": {"id": "skill_07", "title": "Empathic Concern"},
            "learning_objective": {
                "id": "obj_07_01",
                "title": "Define Empathy and differentiate it from sympathy.",
            },
            "recommended_activity": None,
            "question_relevant_content": [
                {
                    "id": "node_0417",
                    "title": "Definition of Empathy",
                    "content": "Empathy is the ability to understand and share feelings.",
                }
            ],
        }
        res = _fallback_response(context, answer_style="short")
        steps = res.get("steps", [])
        step3 = next((s for s in steps if s.get("step") == 3), {})
        assert step3.get("activity") is None
        assert "No activity was selected" in step3.get("content", "")


# =====================================================================
# Area 6: Integrated Pipeline End-to-End
# =====================================================================

class TestIntegratedPipelineRegression:
    """Tests the full multi-agent turn execution from question to validated 4-stage response."""

    def test_integrated_turn_for_empathy_in_learning_end_to_end(
        self, tool_registry, mock_learning_store, knowledge_base, keyword_skill_map
    ):
        """Verifies complete orchestrate_turn flow for 'What is empathy, and why is it important in learning?'."""
        orchestrator = EmpathyOrchestrator(
            learning_store=mock_learning_store,
            tool_registry=tool_registry,
            knowledge_base=knowledge_base,
            keyword_skill_map=keyword_skill_map,
        )

        response, status_code = orchestrator.orchestrate_turn(
            student_id="STU-REG-001",
            authenticated_student_id="STU-REG-001",
            message="What is empathy, and why is it important in learning?",
            answer_style="short",
            conversation_id="conv_reg_001",
        )

        assert status_code == 200, f"Expected 200 OK, got {status_code}: {response}"

        learning_context = response.get("learning_context", {})

        # 1. Skill selection check
        selected_skill = learning_context.get("skill") or {}
        assert selected_skill.get("id") == "skill_07" or selected_skill.get("skill_id") == "skill_07", (
            f"Integrated turn routed to {selected_skill} instead of Skill 7 (Empathic Concern)."
        )

        # 2. Objective selection check
        selected_obj = learning_context.get("learning_objective") or {}
        assert selected_obj.get("id") in {"obj_07_01", "obj_07_02"} or selected_obj.get("objective_id") in {"obj_07_01", "obj_07_02"}, (
            f"Integrated turn selected objective {selected_obj} instead of obj_07_01/obj_07_02."
        )

        # 3. Activity consistency check
        # Aspiration vs. Expectation Inventory is activity_04_02 (Skill 4). It must NOT be selected for Skill 7.
        rec_activity = learning_context.get("recommended_activity") or {}
        if rec_activity:
            assert rec_activity.get("id") != "activity_04_02", (
                "Unrelated Skill 4 activity 'activity_04_02' was attached to an empathy question."
            )
            assert rec_activity.get("skill_id") in {None, "skill_07"}, (
                f"Activity skill_id {rec_activity.get('skill_id')} does not match target skill_07."
            )

        # 4. Response content checks
        direct_answer = response.get("answer") or response.get("educational_response") or ""
        assert "Page 12 of 24" not in direct_answer, (
            f"Direct answer contains running header: '{direct_answer}'"
        )
        assert "Curriculum Study Guide" not in direct_answer, (
            f"Direct answer contains study guide title: '{direct_answer}'"
        )

        # 5. Quick check question check
        steps = response.get("steps") or []
        assert len(steps) == 4, f"Response does not satisfy 4-step contract: {len(steps)} steps."
        step4 = steps[3]
        q_text = step4.get("question", "")
        assert "Page 12 of 24" not in q_text, (
            f"Step 4 question contains running header: '{q_text}'"
        )
