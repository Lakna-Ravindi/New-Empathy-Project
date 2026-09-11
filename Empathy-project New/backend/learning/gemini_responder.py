import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / "backend" / ".env")

SYSTEM_INSTRUCTION = """
You are the language-response component of an educational empathy application.

The Pedagogical Controller has already selected the approved empathy skill,
learning objective, and activity. Treat this structured context as authoritative.

Your primary responsibility is to answer the student's actual question directly
and practically. Use the approved skill, learning objective, and activity to
support and enrich the answer, but do not let them replace or overshadow the
student's immediate question or situation.

Do not select another skill, objective, activity, or learning path.
Do not invent curriculum content. Do not diagnose or provide medical advice.

The discussion must follow four learning steps:

1. Directly answer the student's question.
2. Explain the approved learning objective.
3. Introduce the approved activity.
4. Provide a simple Yes/No or True/False practice question.

Generate all four steps in the response, but keep them clearly separated.
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
The response must contain exactly four steps.

Return valid JSON only, in exactly this shape:
{
  "steps": [
     {"step": 1, "type": "direct_answer", "content": "..."},
     {"step": 2, "type": "learning_objective", "content": "..."},
     {"step": 3, "type": "activity", "content": "...", "activity": {"id": "...", "title": "...", "type": "..."}},
     {"step": 4, "type": "practice_check", "content": "...", "question_type": "true_false", "question": "..."}
  ]
}

The student's situation should remain the main focus throughout the response.
Avoid long theoretical explanations, generic educational content, or responses
that focus mainly on the learning objective instead of the student's question.
"""


def _parse_steps(response_text: str) -> list:
    """Parse and validate Gemini's four-step response contract."""

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
    if not isinstance(activity, dict):
        raise ValueError("The activity step must include activity metadata.")
    for field in ("id", "title", "type"):
        if not isinstance(activity.get(field), str) or not activity[field].strip():
            raise ValueError(f"Activity metadata '{field}' is missing.")

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

    return steps


def generate_educational_response(learning_context: dict) -> list:
    """Turn a controller decision into student-facing language."""

    if learning_context["status"] != "learning_path_selected":
        return learning_context["message"]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

    prompt = {
        "student_question": learning_context["student_question"],
        "selected_skill": learning_context["skill"],
        "selected_topic": learning_context["topic"],
        "selected_learning_objective": learning_context["learning_objective"],
        "selected_activity": learning_context["recommended_activity"],
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

            print("Gemini finish reason:", response.candidates[0].finish_reason)

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            return _parse_steps(response.text)

        except ServerError as e:
            print(f"Gemini 503 error - attempt {attempt + 1}/3")

            if attempt < 2:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Gemini JSON/validation error - attempt {attempt + 1}/3")
            print(f"Error: {e}")

            if attempt < 2:
                wait_time = 1 + attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
