import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / "backend" / ".env")

SYSTEM_INSTRUCTION = """
You are the language-response component of an educational empathy application.

The Pedagogical Controller has already selected the approved empathy skill,
learning objective, and activity. Treat this structured context as authoritative.
The question_relevant_content field contains authoritative approved SEEK content
chunks selected for the student's question. Prioritize those chunks when
answering, but do not invent curriculum facts or attribute unsupported details
to SEEK.

Your primary responsibility is to answer the student's actual question directly
and practically. Use the approved skill, learning objective, and activity to
support and enrich the answer, but do not let them replace or overshadow the
student's immediate question or situation.
When the student has explicitly asked how to cope or asked for help, do not
respond by asking whether they want help. Give the useful answer first; an
approved activity may be offered afterward as an optional next step.

Do not select another skill, objective, activity, or learning path.
Do not invent curriculum content. Do not diagnose or provide medical advice.

The discussion must follow four learning steps:

1. Directly answer the student's question in the top-level "answer" field.
2. Explain the approved learning objective.
3. Introduce the approved activity when one was selected. If no approved
   activity was selected, use this step to describe the next learning idea
   without an activity object. Never invent activity metadata.
4. Provide a simple Yes/No or True/False practice question.

Generate a complete main answer separately from the four supporting steps.
The answer must directly address the student's question and must not depend on
the frontend revealing later steps.
The frontend will present these steps to the student one at a time.
Do not combine the four steps into one paragraph.
The practice question must be a Yes/No or True/False question.
Do not use an open-ended reflection question.

IMPORTANT JSON RULES:

Return ONLY valid JSON.
Do not return Markdown.
Do not wrap the JSON in ```json or ```.
All JSON strings must be properly closed.
Do not place raw line breaks inside string values.
Escape quotation marks inside strings when necessary.
Do not stop generating before all four steps are complete.
The response must contain a non-empty "answer" and exactly four steps.

Return valid JSON only, in exactly this shape:
{
  "answer": "...",
  "steps": [
     {"step": 1, "type": "direct_answer", "content": "..."},
     {"step": 2, "type": "learning_objective", "content": "..."},
     {"step": 3, "type": "activity", "content": "...", "activity": null},
     {"step": 4, "type": "practice_check", "content": "...", "question_type": "true_false", "question": "..."}
  ]
}

The student's situation should remain the main focus throughout the response.
Avoid long theoretical explanations, generic educational content, or responses
that focus mainly on the learning objective instead of the student's question.
"""


def _parse_response(response_text: str, selected_activity=None) -> dict:
    """Parse and validate Gemini's answer and four-step response contract."""

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    print("\n========== GEMINI RAW RESPONSE ==========")
    print(repr(response_text))
    print("==========================================\n")

    cleaned_text = response_text.strip()
    cleaned_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    cleaned_text = re.sub(
        r"\s*```$",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    ).strip()

    print("\n========== CLEANED JSON ==========")
    print(repr(cleaned_text))
    print("==================================\n")

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print("\n========== JSON PARSE ERROR ==========")
        print(f"Message: {e.msg}")
        print(f"Line: {e.lineno}")
        print(f"Column: {e.colno}")
        print(f"Position: {e.pos}")
        print("======================================\n")
        raise ValueError(
            f"Gemini returned invalid JSON: {e.msg} "
            f"(line {e.lineno}, column {e.colno})"
        ) from e

    answer = payload.get("answer") if isinstance(payload, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Gemini response must contain a non-empty answer.")

    steps = payload.get("steps") if isinstance(payload, dict) else None

    if not isinstance(steps, list) or len(steps) != 4:
        raise ValueError("Gemini response must contain exactly four steps.")

    expected_types = [
        "direct_answer",
        "learning_objective",
        "activity",
        "practice_check",
    ]
    for index, (step, expected_type) in enumerate(zip(steps, expected_types), start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} must be an object.")
        if step.get("step") != index:
            raise ValueError(f"Expected step number {index}, got {step.get('step')}")
        if step.get("type") != expected_type:
            raise ValueError(
                f"Expected step type '{expected_type}', got '{step.get('type')}'"
            )
        if not isinstance(step.get("content"), str) or not step["content"].strip():
            raise ValueError(f"Step {index} must contain content.")

    activity = steps[2].get("activity")
    selected_activity = _valid_activity_metadata(selected_activity)
    if selected_activity:
        if not isinstance(activity, dict):
            raise ValueError("The selected activity must include activity metadata.")
        for field in ("id", "title", "type"):
            if not isinstance(activity.get(field), str) or not activity[field].strip():
                raise ValueError(f"Activity metadata '{field}' is missing.")
        for field in ("id", "title", "type"):
            if activity[field] != selected_activity.get(field):
                raise ValueError(
                    f"Gemini activity metadata '{field}' does not match "
                    "the controller-selected activity."
                )
    elif activity is not None:
        raise ValueError(
            "The activity step must use null activity metadata when no "
            "approved activity was selected."
        )

    practice_step = steps[3]
    if practice_step.get("question_type") not in {"true_false", "yes_no"}:
        raise ValueError(
            "The final Gemini step must be a Yes/No or True/False check."
        )
    if (
        not isinstance(practice_step.get("question"), str)
        or not practice_step["question"].strip()
    ):
        raise ValueError("The final Gemini step must contain a question.")

    return {"answer": answer, "steps": steps}


def _valid_activity_metadata(activity):
    """Return selectable activity metadata, or None for an unavailable activity."""
    if not isinstance(activity, dict):
        return None
    fields = ("id", "title", "type")
    if not all(isinstance(activity.get(field), str) and activity[field].strip()
               for field in fields):
        return None
    return {field: activity[field] for field in fields}


def _fallback_response(learning_context: dict, answer_style: str = "short") -> dict:
    """Keep the learning flow usable when Gemini is temporarily unavailable."""

    objective = learning_context.get("learning_objective") or {}
    activity = _valid_activity_metadata(
        learning_context.get("recommended_activity")
    )
    skill = learning_context.get("skill") or {}
    question = learning_context.get("student_question", "")
    objective_title = objective.get("title") or "emotional regulation"
    relevant_content = learning_context.get("question_relevant_content") or []
    source_excerpt = next(
        (
            item.get("content", "").strip()
            for item in relevant_content
            if isinstance(item, dict) and item.get("content")
        ),
        "",
    )

    if answer_style == "long":
        answer = (
            f"Direct answer to your question ({question}): "
            f"{source_excerpt or 'Use the approved learning objective as your next step.'} "
            f"This is connected to {objective_title}. 1) Review the approved "
            "learning idea. 2) Consider how it applies to your situation."
        )
    else:
        answer = (
            f"You asked: {question} "
            f"{source_excerpt or 'Start with the approved learning objective.'}"
        )

    activity_content = (
        f"Try the recommended activity: {activity['title']}."
        if activity
        else "No suitable approved activity was found for this question. Continue with the learning objective above."
    )

    return {
        "answer": answer,
        "steps": [
        {
            "step": 1,
            "type": "direct_answer",
            "content": answer,
        },
        {
            "step": 2,
            "type": "learning_objective",
            "content": (
                f"Learning objective for {skill.get('title') or 'your empathy skill'}: "
                f"{objective_title}. Notice the feeling, connect it to what happened, "
                "and choose a response that supports you and others."
            ),
        },
        {
            "step": 3,
            "type": "activity",
            "content": activity_content,
            "activity": activity,
        },
        {
            "step": 4,
            "type": "practice_check",
            "content": "Before continuing, check in with yourself.",
            "question_type": "true_false",
            "question": "Taking a short pause can help you notice your emotional state.",
        },
        ],
    }


def generate_educational_response(
    learning_context: dict,
    answer_style: str = "short",
) -> dict:
    """Turn a controller decision into student-facing language."""

    if learning_context["status"] != "learning_path_selected":
        return learning_context["message"]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Gemini configuration is unavailable; using fallback response.")
        return _fallback_response(learning_context, answer_style)

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    style_instructions = {
        "short": (
            "Write the top-level answer in 1-2 clear sentences using simple, "
            "direct language. Keep each supporting step concise."
        ),
        "long": (
            "Write a structured educational answer of about 120-180 words. Start "
            "with a direct answer, explain the relevant approved concept, include "
            "one relevant example, and provide numbered practical steps or an "
            "exercise. Include the supported learning objective and activity in "
            "their dedicated steps, then end with a short understanding-check "
            "question. Use only the approved SEEK context below."
        ),
    }

    prompt = {
        "student_question": learning_context["student_question"],
        "answer_style": answer_style,
        "response_style_instructions": style_instructions[answer_style],
        "selected_skill": learning_context["skill"],
        "selected_topic": learning_context["topic"],
        "selected_learning_objective": learning_context["learning_objective"],
        "selected_activity": learning_context["recommended_activity"],
        "question_relevant_content": learning_context.get(
            "question_relevant_content", []
        ),
        "matched_skills": learning_context.get("matched_skills", []),
        "matched_keywords": learning_context.get("matched_keywords", []),
        "source_page": learning_context["source_page"],
    }

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=json.dumps(prompt, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )

            finish_reason = (
                response.candidates[0].finish_reason
                if getattr(response, "candidates", None)
                else None
            )
            print("Gemini finish reason:", finish_reason)
            if str(finish_reason).upper().endswith("MAX_TOKENS"):
                raise ValueError("Gemini response was truncated before completion.")

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            return _parse_response(
                response.text,
                learning_context.get("recommended_activity"),
            )

        except ServerError as e:
            print(f"Gemini 503 error - attempt {attempt + 1}/3")

            if attempt < 2:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Gemini unavailable after retries: {e}")
                return _fallback_response(learning_context, answer_style)

        except ClientError as e:
            if getattr(e, "code", None) == 429:
                print(f"Gemini quota exhausted; using fallback response: {e}")
                return _fallback_response(learning_context, answer_style)
            print(f"Gemini client error; using fallback response: {e}")
            return _fallback_response(learning_context, answer_style)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Gemini JSON/validation error - attempt {attempt + 1}/3")
            print(f"Error: {e}")

            if attempt < 2:
                wait_time = 1 + attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Gemini response remained invalid after retries; using fallback.")
                return _fallback_response(learning_context, answer_style)
