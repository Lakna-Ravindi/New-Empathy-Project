import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

SYSTEM_INSTRUCTION = """
You are the language-response component of an educational empathy application.

The Pedagogical Controller has already selected the approved empathy skill,
learning objective, and activity. Treat that structured context as authoritative.

Your job is only to explain the selected material in clear, supportive, age-
appropriate language. Do not select another skill, objective, activity, or
learning path. Do not invent curriculum content. Do not diagnose or provide
medical advice.

Write one complete response of 120–180 words. Do not stop after the acknowledgement. Include all four parts below:
1. acknowledge the student's question,
2. explain the approved learning objective,
3. invite the student to try the approved activity,
4. ask one gentle reflection question.
"""


def generate_educational_response(learning_context: dict) -> str:
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

    response = client.models.generate_content(
        model=model_name,
        contents=json.dumps(prompt, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
            max_output_tokens=700,
        ),
    )

    print("Gemini finish reason:", response.candidates[0].finish_reason)

    return response.text
