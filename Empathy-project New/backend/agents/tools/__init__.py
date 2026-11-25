"""Controlled Tool Execution Layer for Empathy Guru.

Exports the 9 controlled tool functions, the RBAC ToolRegistry, and exception types.
"""

from agents.tools.curriculum_tools import (
    get_learning_activity,
    retrieve_approved_seek_content,
)
from agents.tools.evaluation_tools import (
    evaluate_student_response,
    save_learning_interaction,
)
from agents.tools.progress_tools import (
    get_current_skill,
    get_pending_objectives,
    get_student_progress,
    update_student_progress,
    validate_activity_completion,
)
from agents.tools.registry import (
    ToolError,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistry,
    ToolValidationError,
)

__all__ = [
    # 9 Required Tool Functions
    "retrieve_approved_seek_content",
    "get_student_progress",
    "get_pending_objectives",
    "get_current_skill",
    "get_learning_activity",
    "evaluate_student_response",
    "validate_activity_completion",
    "save_learning_interaction",
    "update_student_progress",
    # Registry & Exceptions
    "ToolRegistry",
    "ToolError",
    "ToolPermissionError",
    "ToolValidationError",
    "ToolExecutionError",
    "get_default_tool_registry",
]


def get_default_tool_registry() -> ToolRegistry:
    """Constructs and returns the canonical ToolRegistry with all 9 tools and RBAC rules."""
    registry = ToolRegistry()

    # 1. retrieve_approved_seek_content
    # Permitted: learning_planner_agent, tutor_agent, orchestrator
    registry.register_tool(
        name="retrieve_approved_seek_content",
        func=retrieve_approved_seek_content,
        allowed_roles={"planner", "learning_planner_agent", "tutor", "tutor_agent", "orchestrator", "system_admin"},
        description="Retrieves approved UNESCO SEEK curriculum content nodes using semantic search.",
    )

    # 2. get_student_progress
    # Permitted: learning_planner_agent, assessment_agent, orchestrator
    registry.register_tool(
        name="get_student_progress",
        func=get_student_progress,
        allowed_roles={"planner", "learning_planner_agent", "assessment", "assessment_agent", "orchestrator", "system_admin"},
        description="Returns student curriculum progress summary across all 8 skills (read-only).",
    )

    # 3. get_pending_objectives
    # Permitted: learning_planner_agent, assessment_agent, orchestrator
    registry.register_tool(
        name="get_pending_objectives",
        func=get_pending_objectives,
        allowed_roles={"planner", "learning_planner_agent", "assessment", "assessment_agent", "orchestrator", "system_admin"},
        description="Returns uncompleted canonical learning objectives for the student (read-only).",
    )

    # 4. get_current_skill
    # Permitted: learning_planner_agent, orchestrator
    registry.register_tool(
        name="get_current_skill",
        func=get_current_skill,
        allowed_roles={"planner", "learning_planner_agent", "orchestrator", "system_admin"},
        description="Determines the student's active empathy skill in curriculum progression (read-only).",
    )

    # 5. get_learning_activity
    # Permitted: learning_planner_agent, tutor_agent, orchestrator
    registry.register_tool(
        name="get_learning_activity",
        func=get_learning_activity,
        allowed_roles={"planner", "learning_planner_agent", "tutor", "tutor_agent", "orchestrator", "system_admin"},
        description="Retrieves an approved SEEK activity without inventing new ones (read-only).",
    )

    # 6. evaluate_student_response
    # Permitted: assessment_agent, orchestrator
    registry.register_tool(
        name="evaluate_student_response",
        func=evaluate_student_response,
        allowed_roles={"assessment", "assessment_agent", "orchestrator", "system_admin"},
        description="Evaluates student reflections, practice checks, or quiz submissions (read-only).",
    )

    # 7. validate_activity_completion
    # Permitted: assessment_agent, orchestrator
    registry.register_tool(
        name="validate_activity_completion",
        func=validate_activity_completion,
        allowed_roles={"assessment", "assessment_agent", "orchestrator", "system_admin"},
        description="Validates that an activity exists in the curriculum and matches required types (read-only).",
    )

    # 8. save_learning_interaction
    # Permitted: tutor_agent, support_agent, orchestrator, system_admin
    registry.register_tool(
        name="save_learning_interaction",
        func=save_learning_interaction,
        allowed_roles={"orchestrator", "tutor_agent", "support_agent", "system_admin"},
        description="Persists an educational learning interaction to the database.",
    )

    # 9. update_student_progress
    # Permitted: strictly assessment_agent and system_admin only
    registry.register_tool(
        name="update_student_progress",
        func=update_student_progress,
        allowed_roles={"assessment", "assessment_agent", "system_admin"},
        description="Controlled mutation of student progress with verification and duplicate prevention.",
    )

    return registry
