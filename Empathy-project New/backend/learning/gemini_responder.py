import json
import os
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

Write one complete response of 120–180 words. Do not stop after acknowledging
the student's question.

Structure the response naturally using these priorities:

1. Directly acknowledge and answer the student's actual question with
   practical, empathetic guidance relevant to their situation.
2. Briefly connect the advice to the approved learning objective. Explain the
   concept only as much as needed to help the student understand or apply it.
3. Naturally introduce and invite the student to try the approved activity
   when it is relevant to their situation.
4. End with one gentle reflection or practice question that encourages the
   student to apply what they have learned.

The student's situation should remain the main focus throughout the response.
Avoid long theoretical explanations, generic educational content, or responses
that focus mainly on the learning objective instead of the student's question.
"""


def generate_educational_response(learning_context: dict) -> str:
    """Turn a controller decision into student-facing language."""

    if learning_context["status"] != "learning_path_selected":
        return learning_context["message"]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.getenv("GEMINI_MODEL") or "gemini-3.6-flash"

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
                ),
            )

            print("Gemini finish reason:", response.candidates[0].finish_reason)
            return response.text

        except ServerError as e:
            print(f"Gemini 503 error - attempt {attempt + 1}/3")

            if attempt < 2:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
