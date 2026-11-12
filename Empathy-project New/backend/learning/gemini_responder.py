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
learning objective, and activity or registered chatbot-practice scaffold. Treat
this structured context as authoritative.
The question_relevant_content field contains authoritative approved SEEK content
chunks selected for the student's question. Prioritize those chunks when
answering, but do not invent curriculum facts or attribute unsupported details
to SEEK.
Generate suggested_questions only from the supplied approved SEEK content for
the suggestion_learning_objective and suggestion_relevant_content fields. If
there is not enough supplied content to form relevant questions, return an
empty list.

Your primary responsibility is to answer the student's actual question directly
and practically. Use the approved skill, learning objective, and activity to
support and enrich the answer, but do not let them replace or overshadow the
student's immediate question or situation.
When the student has explicitly asked how to cope or asked for help, do not
respond by asking whether they want help. Give the useful answer first; an
approved activity may be offered afterward as an optional next step.

Do not select another skill, objective, activity, or learning path.
Do not invent curriculum content. Never diagnose or provide medical advice.
If the student describes potentially serious physical symptoms, including
severe chest tightness or difficulty breathing, prioritize immediate safety and
advise seeking urgent medical assistance. Do not assume those symptoms are
caused by stress.

Structure the conversation using these four stages, in order. The frontend
reveals the stage content progressively across conversation turns; do not write
or present all four stages as one student-facing response. Keep each generated
stage focused, concise, and appropriate to the student's latest message:

1. Direct response: Answer the student's actual question first. Acknowledge
   their feelings naturally and give practical, empathetic guidance for their
   immediate situation. Never begin with "The approved SEEK content says" or
   copy curriculum text directly as the answer.
2. Explain the concept: After addressing the immediate concern, briefly explain
   the relevant approved SEEK concept in simple, natural language. Connect it
   to the student's situation with a practical example when useful.
3. Guided activity: Introduce only the activity selected by the controller.
   For a registered chatbot-practice scaffold, create one short, optional
   student-facing scenario or reflection prompt grounded in the supplied
   approved SEEK content. Do not claim that chatbot-created practice came from
   the PDF. Match the interaction to activity.interaction_type: practical
   activities give an exercise and ask for a completion confirmation;
   reflective activities ask the student to type a brief reflection; combined
   activities give an exercise, ask for a brief reflection, and require the
   student to confirm completing the exercise. For a PDF activity, use its
   supplied instructions. If no activity was selected, say so without
   inventing an alternative.
4. Understanding check: Only after the student has had an opportunity to
   understand and practise the concept, ask one relevant, simple question.
   Use either a True/False check or a multiple-choice check. For multiple
   choice, include three or four concise options and return the correct option
   text separately in "correct_answer". Do not put the answer in the question
   itself. Do not ask whether the learning objective is relevant.

Conversation rules:
- Treat the student's latest message as the immediate priority and respond to
  it before advancing to another stage.
- Do not force every stage into every response. Do not repeat explanations
  unnecessarily or restart the learning flow after each message.
- If the student is confused, explain the same approved concept more simply
  and check understanding before proceeding.
- Use warm, conversational English. Avoid robotic introductions, textbook-like
  paragraphs, unnecessary headings, and "The approved SEEK content says".
- Use only information supported by the supplied approved SEEK content for
  curriculum explanations. Do not select a different skill, objective,
  activity, or learning path. A registered chatbot-practice scaffold may be
  phrased as a new short exercise, but must not be described as PDF content.

IMPORTANT JSON RULES:

Return ONLY valid JSON.
Do not return Markdown.
Do not wrap the JSON in ```json or ```.
All JSON strings must be properly closed.
Do not place raw line breaks inside string values.
Escape quotation marks inside strings when necessary.
Return the four stage fields in the structured payload so the frontend can
reveal them one at a time. A stage field is not permission to present that
stage early in the visible conversation. The response must contain a non-empty
"answer" and exactly four ordered steps.
When an activity is selected, copy its id, title, type, and interaction_type
exactly into the step 3 activity object. Do not invent or alter activity
metadata. Use null only when the controller supplied no activity.
Include 3 or 4 short, natural, non-empty suggested questions when the supplied
approved content supports them. Do not suggest unrelated counselling,
medical, or crisis questions.

Return valid JSON only, in exactly this shape:
{
  "answer": "...",
  "suggested_questions": ["...", "...", "..."],
  "steps": [
     {"step": 1, "type": "direct_answer", "content": "..."},
     {"step": 2, "type": "learning_objective", "content": "..."},
     {"step": 3, "type": "activity", "content": "...", "activity": null},
     {"step": 4, "type": "practice_check", "content": "...", "question_type": "multiple_choice", "question": "...", "options": ["...", "...", "..."], "correct_answer": "..."}
  ]
}

The student's situation should remain the main focus throughout the response.
Avoid long theoretical explanations, generic educational content, or responses
that focus mainly on the learning objective instead of the student's question.

For both short and long answer styles:
- Use simple, clear, student-friendly English. Explain difficult concepts
  in everyday language without changing their original meaning.
- Correct grammar, spelling, punctuation, and awkward sentence structures.
- Improve the logical flow by organizing ideas clearly and using natural
  transitions between related points.
- Avoid unnecessary repetition, overly complex vocabulary, and excessively
  long sentences.
- Keep the answer appropriate to the current stage and the student's latest
  message. "Long" may add useful detail to the current stage, but must not
  combine multiple stages or become a complete lesson.
- Where relevant, include one short, realistic example from everyday
  Sri Lankan student life, such as A/L examination stress, university
  group projects, classroom discussions, or misunderstandings between friends.
- Examples must be relevant to the student's actual question and should
  help the student understand the concept practically.
- Do not force an example into an answer when it is not useful.
- Never include religious practices, rituals, teachings, scriptures,
  religious figures, or religious activities as examples, explanations,
  or recommendations. Use neutral, inclusive, everyday Sri Lankan contexts.
- Do not present illustrative examples as facts from the approved SEEK
  curriculum.
- Keep all explanations grounded in the supplied approved SEEK content.
  Never invent curriculum facts, objectives, activities, or instructions.
"""


def _parse_response(
    response_text: str,
    selected_activity=None,
    allow_empty_suggestions: bool = False,
) -> dict:
    """Parse and validate Gemini's answer, suggestions, and four-step contract."""

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

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

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Gemini returned invalid JSON."
        ) from e

    answer = payload.get("answer") if isinstance(payload, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Gemini response must contain a non-empty answer.")
    if _has_repeated_paragraph(answer) or _looks_truncated(answer):
        raise ValueError("Gemini answer was duplicated or truncated.")

    suggested_questions = (
        payload.get("suggested_questions") if isinstance(payload, dict) else None
    )
    if not isinstance(suggested_questions, list) or not (
        (allow_empty_suggestions and len(suggested_questions) == 0)
        or 3 <= len(suggested_questions) <= 4
    ):
        raise ValueError(
            "Gemini response must contain three or four non-empty suggested "
            "questions, or an empty list when approved suggestion content is "
            "unavailable."
        )
    if any(
        not isinstance(question, str) or not question.strip()
        for question in suggested_questions
    ):
        raise ValueError(
            "Gemini response must contain non-empty suggested questions."
        )
    suggested_questions = [question.strip() for question in suggested_questions]
    if len({question.casefold() for question in suggested_questions}) != len(
        suggested_questions
    ):
        raise ValueError("Gemini suggested questions must be distinct.")

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
        if (
            _has_repeated_paragraph(step["content"])
            or _looks_truncated(step["content"])
        ):
            raise ValueError(f"Step {index} was duplicated or truncated.")
        if index == 3 and _contains_internal_activity_instruction(
            step["content"]
        ):
            raise ValueError("Gemini exposed an internal activity instruction.")

    activity = steps[2].get("activity")
    selected_activity = _valid_activity_metadata(selected_activity)
    if selected_activity:
        if not isinstance(activity, dict):
            raise ValueError("The selected activity must include activity metadata.")
        activity = _valid_activity_metadata(activity)
        if not activity:
            raise ValueError("The selected activity metadata is invalid.")
        for field in ("id", "title", "type", "interaction_type"):
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
    question_type = str(practice_step.get("question_type") or "").strip().lower()
    normalized_question_type = re.sub(r"\s+|[-_]", "_", question_type)
    normalized_question_type = normalized_question_type.replace("yesno", "yes_no")
    normalized_question_type = normalized_question_type.replace("truefalse", "true_false")
    if normalized_question_type in {"true_false", "yes_no", "multiple_choice"}:
        practice_step["question_type"] = normalized_question_type
    else:
        raise ValueError(
            "The final Gemini step must be a Yes/No, True/False, or "
            "multiple-choice check."
        )
    if (
        not isinstance(practice_step.get("question"), str)
        or not practice_step["question"].strip()
    ):
        raise ValueError("The final Gemini step must contain a question.")

    correct_answer = practice_step.get("correct_answer")
    if normalized_question_type == "multiple_choice":
        options = practice_step.get("options")
        if (
            not isinstance(options, list)
            or not 3 <= len(options) <= 4
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or len({option.strip().casefold() for option in options}) != len(options)
            or not isinstance(correct_answer, str)
            or not correct_answer.strip()
            or correct_answer.strip().casefold()
            not in {option.strip().casefold() for option in options}
        ):
            raise ValueError(
                "A multiple-choice check must contain three or four distinct "
                "options and a correct answer matching one option."
            )
        practice_step["options"] = [option.strip() for option in options]
        practice_step["correct_answer"] = correct_answer.strip()
    elif not isinstance(correct_answer, bool):
        raise ValueError(
            "The final True/False or Yes/No step must contain a boolean "
            "correct_answer."
        )

    return {
        "answer": answer.strip(),
        "steps": steps,
        "suggested_questions": suggested_questions,
    }


def _deduplicate_content(items):
    """Keep complete, unique approved chunks before sending them to Gemini."""
    result = []
    seen_ids = set()
    seen_text = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        content = str(candidate.get("content") or "").strip()
        source = candidate.get("source")
        source_text = source.get("text") if isinstance(source, dict) else None
        if isinstance(source_text, str) and len(source_text.strip()) > len(content):
            candidate["content"] = source_text.strip()
            content = candidate["content"]
        stable_id = (
            candidate.get("item_id")
            or candidate.get("content_id")
            or candidate.get("id")
        )
        text_key = re.sub(r"\s+", " ", content).casefold()
        if (stable_id and stable_id in seen_ids) or (
            text_key and text_key in seen_text
        ):
            continue
        if stable_id:
            seen_ids.add(stable_id)
        if text_key:
            seen_text.add(text_key)
        result.append(candidate)
    return result


def _compact_text(text):
    """Remove repeated extracted sentences without inventing replacement text."""
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", str(text or "").strip()))
    unique = []
    seen = set()
    for sentence in sentences:
        key = sentence.casefold().strip()
        if key and key not in seen:
            unique.append(sentence.strip())
            seen.add(key)
    return " ".join(unique)


def _has_repeated_paragraph(text: str) -> bool:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip().casefold()
        for paragraph in str(text).split("\n\n")
        if paragraph.strip()
    ]
    return len(paragraphs) != len(set(paragraphs))


def _looks_truncated(text: str) -> bool:
    return not re.search(r"[.!?…][\"')\]]?$", str(text).strip())


def _contains_internal_activity_instruction(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text).casefold())
    return any(
        phrase in normalized
        for phrase in (
            "create one short",
            "internal instruction",
            "prompt template",
            "generation instruction",
        )
    )


def _strip_internal_activity_labels(text: str) -> str:
    """Keep the complete approved instruction while hiding extraction labels."""
    return re.sub(
        r"^(?:tracking|activity|practice|reflection)\s+instruction\s*:\s*",
        "",
        re.sub(r"\s+", " ", str(text or "")).strip(),
        flags=re.IGNORECASE,
    ).strip()


def _valid_activity_metadata(activity):
    """Return selectable activity metadata, or None for an unavailable activity."""
    if not isinstance(activity, dict):
        return None
    fields = ("id", "title", "type")
    if not all(isinstance(activity.get(field), str) and activity[field].strip()
               for field in fields):
        return None
    interaction_type = activity.get("interaction_type")
    if interaction_type not in {"practical", "reflective", "combined"}:
        interaction_type = (
            "reflective" if activity.get("type") == "reflection" else "practical"
        )
    return {
        **{field: activity[field] for field in fields},
        "interaction_type": interaction_type,
    }


def _mentions_urgent_physical_symptom(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(question or "").casefold())
    urgent_phrases = (
        "severe chest tightness",
        "chest tightness",
        "difficulty breathing",
        "trouble breathing",
        "shortness of breath",
        "can't breathe",
        "cannot breathe",
    )
    return any(phrase in normalized for phrase in urgent_phrases)


def _fallback_suggested_questions(learning_context: dict) -> list[str]:
    """Build questions from the approved content selected for suggestions."""

    objective = (
        learning_context.get("suggestion_learning_objective")
        or learning_context.get("learning_objective")
        or {}
    )
    relevant_content = (
        learning_context.get("suggestion_relevant_content")
        or learning_context.get("question_relevant_content")
        or []
    )
    if not isinstance(relevant_content, list):
        relevant_content = []
    approved_text = " ".join(
        str(item.get("content") or "")
        for item in relevant_content
        if isinstance(item, dict)
    ).casefold()
    if "affective" in approved_text and "cognitive" in approved_text:
        return [
            "What is affective empathy?",
            "What is cognitive empathy?",
            "How are affective and cognitive empathy different?",
            "Can you give me an example of each?",
        ]

    topics = []
    for item in relevant_content:
        if not isinstance(item, dict):
            continue
        title = re.sub(r"\s+", " ", str(item.get("title") or "").strip())
        content = _compact_text(item.get("content") or "")
        topic = (
            title
            if title and len(title) <= 90 and title[-1:] in ".?!:" or
            title and len(title) <= 45
            else content.split(".")[0].strip()
        )
        if not topic:
            topic = title
        if len(topic) > 90:
            topic = topic[:90].rsplit(" ", 1)[0].rstrip(" ,;:")
        if topic and topic.casefold() not in {value.casefold() for value in topics}:
            topics.append(topic)
        if len(topics) == 2:
            break

    if not topics:
        objective_title = re.sub(
            r"\s+",
            " ",
            str(objective.get("title") or "").strip(),
        )
        if objective_title:
            return [
                f"What does {objective_title} mean?",
                f"Why is {objective_title} important?",
                f"How can you apply {objective_title} in daily life?",
            ]
        return []

    primary_topic = topics[0]
    secondary_topic = topics[1] if len(topics) > 1 else primary_topic
    questions = [
        f"What does {primary_topic} mean?",
        f"How could {primary_topic} appear in everyday life?",
        f"How is {secondary_topic} different from it?",
    ]
    if len(topics) > 1:
        questions.append(f"Can you give an example involving {topics[1]}?")

    unique_questions = []
    seen = set()
    for question in questions:
        key = question.casefold()
        if key not in seen:
            unique_questions.append(question)
            seen.add(key)
    return unique_questions[:4]


def _response_uses_other_objective(response: dict, learning_context: dict) -> bool:
    """Reject model output that explicitly switches to another selected objective."""
    selected_objective = (
        learning_context.get("learning_objective")
        or {}
    )
    selected_id = (
        selected_objective.get("objective_id")
        or selected_objective.get("id")
    )
    other_titles = set()
    for recommendation in learning_context.get("matched_skills", []):
        objective = recommendation.get("learning_objective") or {}
        objective_id = objective.get("objective_id") or objective.get("id")
        title = objective.get("title")
        if (
            objective_id
            and objective_id != selected_id
            and isinstance(title, str)
            and title.strip()
        ):
            other_titles.add(title.strip().casefold())

    response_text = json.dumps(response, ensure_ascii=False).casefold()
    return any(title in response_text for title in other_titles)


def _fallback_response(learning_context: dict, answer_style: str = "short") -> dict:
    """Return a grounded response when Gemini is temporarily unavailable."""

    objective = learning_context.get("learning_objective") or {}
    activity = _valid_activity_metadata(
        learning_context.get("recommended_activity")
    )
    objective_title = objective.get("title")
    objective_content = objective.get("content")
    student_question = learning_context.get("student_question", "")
    question_content = _deduplicate_content(
        learning_context.get("question_relevant_content") or []
    )
    related_topic = next(
        (
            str(item.get("content") or item.get("title") or "").strip()
            for item in question_content
            if isinstance(item, dict)
            and (
                isinstance(item.get("title"), str) and item["title"].strip()
                or isinstance(item.get("content"), str) and item["content"].strip()
            )
        ),
        "",
    )
    related_topic = _compact_text(related_topic)
    if related_topic and len(related_topic) > 260:
        related_topic = related_topic[:260].rsplit(" ", 1)[0].rstrip(" ,;:")
    selected_activity = learning_context.get("recommended_activity") or {}
    activity = _valid_activity_metadata(selected_activity)
    practice_question = bool(
        re.search(
            r"\b(?:how can i|how do i|how to|practice|practise|apply|try)\b",
            student_question.casefold(),
        )
    )
    approved_activity_content = _strip_internal_activity_labels(
        selected_activity.get("content")
    )
    if _mentions_urgent_physical_symptom(student_question):
        answer = (
            "I'm sorry you're experiencing this. Severe chest tightness or "
            "difficulty breathing can be urgent. Please seek immediate medical "
            "help now or contact local emergency services, and tell a trusted "
            "person nearby if you can. I can't determine the cause here."
        )
    elif practice_question and activity and approved_activity_content:
        answer = approved_activity_content
    elif related_topic:
        answer = related_topic
    else:
        answer = (
            "That sounds difficult. I don't have enough approved information "
            "to guide you reliably yet. What part would you most like help with?"
        )

    if activity:
        raw_activity_content = str(selected_activity.get("content") or "").strip()
        if selected_activity.get("source") == "chatbot_generated":
            activity_content = (
                "Imagine you are getting ready for an exam and notice your "
                "shoulders feel tense. Name the body sensation, then consider "
                "which feeling it might help you notice. You can use this "
                "example instead of sharing a personal experience."
            )
        elif _contains_internal_activity_instruction(raw_activity_content):
            activity_content = (
                f"Try the “{activity['title']}” exercise and confirm "
                "when you have finished."
            )
        else:
            activity_content = _strip_internal_activity_labels(raw_activity_content)
        if not activity_content:
            activity_content = (
                f"The activity “{activity['title']}” was selected, but its "
                "instructions are not available here."
            )
    else:
        activity_content = "No activity was selected for this response."

    if question_content:
        objective_step_content = _compact_text(
            str(question_content[0].get("content") or "").strip()
        )
    elif objective_content:
        objective_step_content = _compact_text(objective_content)
    elif objective_title:
        objective_step_content = f"Approved learning objective: {objective_title}."
    else:
        objective_step_content = (
            "No approved learning objective was available for this response."
        )

    taught_text = " ".join(
        str(item.get("content") or "")
        for item in question_content
        if isinstance(item, dict)
    ).casefold()
    if "beneficial mental state" in taught_text:
        practice_question = (
            "True or false: Beneficial mental states are described as leading "
            "toward happiness and away from suffering."
        )
    elif "body literacy" in taught_text:
        practice_question = (
            "True or false: Body literacy involves noticing physical sensations "
            "to better understand emotional states."
        )
    elif "self-compassion" in taught_text or "self compassion" in taught_text:
        practice_question = (
            "True or false: Self-compassion includes recognizing your worth "
            "and supporting your well-being."
        )
    elif "affective empathy" in taught_text and "cognitive empathy" in taught_text:
        practice_question = (
            "True or false: Affective empathy involves sharing another person's "
            "feelings, while cognitive empathy involves understanding their "
            "perspective."
        )
    else:
        taught_statement = _compact_text(
            str(question_content[0].get("content") or "")
            if question_content
            else ""
        )
        taught_statement = taught_statement.split(".")[0].strip()
        practice_question = (
            f"True or false: {taught_statement}."
            if taught_statement
            else "True or false: This concept is described in the lesson."
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
            "content": objective_step_content,
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
            "question": practice_question,
            "correct_answer": bool(objective_title),
        },
        ],
        "suggested_questions": _fallback_suggested_questions(learning_context),
    }


def generate_educational_response(
    learning_context: dict,
    answer_style: str = "short",
) -> dict:
    """Turn a controller decision into student-facing language."""

    if learning_context["status"] != "learning_path_selected":
        return learning_context["message"]

    question_relevant_content = _deduplicate_content(
        learning_context.get("question_relevant_content") or []
    )
    if not any(
        isinstance(item, dict)
        and isinstance(item.get("content"), str)
        and item["content"].strip()
        for item in question_relevant_content
    ):
        print("Relevant approved SEEK content is unavailable; using safe fallback.")
        return _fallback_response(learning_context, answer_style)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Gemini configuration is unavailable; using fallback response.")
        return _fallback_response(learning_context, answer_style)

    model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    selected_activity = learning_context.get("recommended_activity")
    if isinstance(selected_activity, dict):
        selected_activity = dict(selected_activity)
        activity_metadata = _valid_activity_metadata(selected_activity)
        if activity_metadata:
            selected_activity["interaction_type"] = activity_metadata[
                "interaction_type"
            ]
    style_instructions = {
        "short": (
            "Use concise, direct language for the current stage. Do not combine "
            "stages or add a complete lesson."
        ),
        "long": (
            "Use a little more helpful detail within the current stage, while "
            "remaining concise and conversational. Do not combine stages or "
            "write a complete lesson."
        ),
    }

    suggestion_content = prompt_suggestion_content = _deduplicate_content(
        learning_context.get(
            "suggestion_relevant_content",
            question_relevant_content,
        )
    )
    question_content_keys = {
        re.sub(r"\s+", " ", str(item.get("content") or "")).casefold()
        for item in question_relevant_content
        if isinstance(item, dict) and item.get("content")
    }
    if suggestion_content and all(
        re.sub(r"\s+", " ", str(item.get("content") or "")).casefold()
        in question_content_keys
        for item in suggestion_content
    ):
        prompt_suggestion_content = [
            {
                key: item[key]
                for key in ("id", "title", "page")
                if item.get(key) is not None
            }
            for item in suggestion_content
        ]

    prompt = {
        "student_question": learning_context["student_question"],
        "answer_style": answer_style,
        "response_style_instructions": style_instructions[answer_style],
        "selected_skill": learning_context["skill"],
        "selected_topic": learning_context.get("topic"),
        "selected_learning_objective": {
            "id": (learning_context.get("learning_objective") or {}).get("id"),
            "objective_id": (
                learning_context.get("learning_objective") or {}
            ).get("objective_id"),
            "title": (
                learning_context.get("learning_objective") or {}
            ).get("title"),
        },
        "selected_activity": selected_activity,
        "question_relevant_content": question_relevant_content,
        "suggestion_learning_objective": learning_context.get(
            "suggestion_learning_objective",
            learning_context.get("learning_objective"),
        ),
        "suggestion_relevant_content": prompt_suggestion_content,
        "matched_skills": learning_context.get("matched_skills", []),
        "matched_keywords": learning_context.get("matched_keywords", []),
        "source_page": learning_context.get("source_page"),
    }
    allow_empty_suggestions = not _fallback_suggested_questions({
        "suggestion_learning_objective": prompt["suggestion_learning_objective"],
        "suggestion_relevant_content": prompt["suggestion_relevant_content"],
    })

    client = None
    for attempt in range(3):
        try:
            if client is None:
                client = genai.Client(api_key=api_key)
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
            if str(finish_reason).upper().endswith("MAX_TOKENS"):
                raise ValueError("Gemini response was truncated before completion.")

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            parsed_response = _parse_response(
                response.text,
                learning_context.get("recommended_activity"),
                allow_empty_suggestions=allow_empty_suggestions,
            )
            if _response_uses_other_objective(parsed_response, learning_context):
                raise ValueError(
                    "Gemini response referenced an objective other than the selected objective."
                )
            return parsed_response

        except ServerError as e:
            print(f"Gemini service error - attempt {attempt + 1}/3.")

            if attempt < 2:
                wait_time = 2 ** attempt
                print(f"Retrying Gemini request in {wait_time} seconds.")
                time.sleep(wait_time)
            else:
                print("Gemini unavailable after retries; using fallback response.")
                return _fallback_response(learning_context, answer_style)

        except ClientError as e:
            if getattr(e, "code", None) == 429:
                print("Gemini rate limit reached; using fallback response.")
                return _fallback_response(learning_context, answer_style)
            print("Gemini client error; using fallback response.")
            return _fallback_response(learning_context, answer_style)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Gemini response validation failed - attempt {attempt + 1}/3.")

            if attempt < 2:
                wait_time = 1 + attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Gemini response remained invalid; using fallback.")
                return _fallback_response(learning_context, answer_style)
