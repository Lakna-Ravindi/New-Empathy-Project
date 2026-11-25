"""Unit tests for General Empathy Agent prompt module."""

import pytest

from agents.prompts import (
    GENERAL_EMPATHY_SYSTEM_INSTRUCTION,
    build_general_empathy_prompt,
    is_no_advice_requested,
)


def test_system_instruction_contains_core_identity():
    assert "General Empathy Agent for Empathy Guru" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "NOT a therapist" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "NOT a diagnostic system" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "NOT a SEEK curriculum tutor" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION


def test_system_instruction_preserves_communication_principles():
    assert "Listen carefully" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "25–80 words" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Avoid robotic clichés" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "at most one gentle, relevant follow-up question" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Sinhala" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Singlish" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION


def test_system_instruction_enforces_boundaries():
    assert "Do NOT make medical or psychiatric diagnoses" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Never claim to provide professional clinical treatment" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Never invent emergency contact details" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Never claim that you have contacted a counsellor" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION


def test_system_instruction_enforces_seek_separation():
    assert "Strict Separation from SEEK Learning" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Never retrieve, quote, or teach SEEK curriculum topics" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Never recommend SEEK learning objectives" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Never claim that a conversation completes a learning objective" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION
    assert "Never output SEEK four-stage educational response formats" in GENERAL_EMPATHY_SYSTEM_INSTRUCTION


def test_is_no_advice_requested_detection():
    assert is_no_advice_requested("I don't want advice right now.") is True
    assert is_no_advice_requested("Please dont give me advice, just listen.") is True
    assert is_no_advice_requested("I just want to vent.") is True
    assert is_no_advice_requested("No advice needed.") is True
    assert is_no_advice_requested("Stop giving advice.") is True
    assert is_no_advice_requested("I feel happy today.") is False
    assert is_no_advice_requested("Can you give me advice on study habits?") is False
    assert is_no_advice_requested("") is False
    assert is_no_advice_requested(None) is False


def test_build_general_empathy_prompt_empty_message_raises():
    with pytest.raises(ValueError, match="message cannot be empty"):
        build_general_empathy_prompt("")

    with pytest.raises(ValueError, match="message cannot be empty"):
        build_general_empathy_prompt("   ")


def test_build_general_empathy_prompt_single_message():
    contents = build_general_empathy_prompt("I had a stressful day at school.")
    assert isinstance(contents, list)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "I had a stressful day at school."


def test_build_general_empathy_prompt_multi_turn_history():
    history = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there. How has your day been?"},
    ]
    contents = build_general_empathy_prompt(
        message="I'm feeling down.",
        history=history,
    )
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello!"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "Hi there. How has your day been?"
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["text"] == "I'm feeling down."


def test_build_general_empathy_prompt_auto_detects_no_advice():
    contents = build_general_empathy_prompt(
        message="I don't want advice, I just need to vent.",
    )
    user_text = contents[0]["parts"][0]["text"]
    assert "[USER PREFERENCE: The student explicitly requested NO ADVICE" in user_text
    assert "I don't want advice, I just need to vent." in user_text


def test_build_general_empathy_prompt_explicit_preferences():
    contents = build_general_empathy_prompt(
        message="Exam is near.",
        preferences={"no_advice_requested": True, "language": "Sinhala"},
    )
    user_text = contents[0]["parts"][0]["text"]
    assert "NO ADVICE" in user_text
    assert "Respond in Sinhala" in user_text


def test_build_general_empathy_prompt_as_string():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    text = build_general_empathy_prompt("Feeling anxious", history=history, as_string=True)
    assert isinstance(text, str)
    assert "User: Hi" in text
    assert "Assistant: Hello" in text
    assert "User: Feeling anxious" in text
