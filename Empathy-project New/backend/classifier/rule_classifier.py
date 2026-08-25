def classify(block):
    """
    Classifies SEEK Learning PDF blocks into knowledge node types
    and returns confidence score.
    """

    text = block.get("text", "").strip().lower()

    font = block.get("font_name", "").lower()

    try:
        font_size = float(block.get("font_size", 0))
    except:
        font_size = 0


    is_bold = "bold" in font

    


    # -----------------------------------
    # Chapter / Skill Detection
    # -----------------------------------

    if text.startswith("skill"):

        return {
            "type": "chapter",
            "confidence": 0.95
        }



    # -----------------------------------
    # Module Detection
    # -----------------------------------

    if (
        text.startswith("module objectives")
        or text.endswith("module objectives")
    ):

        return {
            "type": "module",
            "confidence": 0.95
        }



    # -----------------------------------
    # Learning Objective Detection
    # -----------------------------------

    objective_keywords = [

        "you will be able to",
        "by the end of this skill",
        "learn to",
        "objectives",
        "identify",
        "recognize",
        "understand",
        "define"

    ]


    if any(
        keyword in text
        for keyword in objective_keywords
    ):

        return {
            "type": "learning_objective",
            "confidence": 0.85
        }



    # -----------------------------------
    # Reflection Detection
    # -----------------------------------

    if any(
        word in text
        for word in [
            "reflect",
            "reflection",
            "questions for reflection",
            "think about"
        ]
    ):

        return {
            "type": "reflection",
            "confidence": 0.90
        }



    # -----------------------------------
    # Assessment Detection
    # -----------------------------------

    if any(
        word in text
        for word in [
            "quiz",
            "assessment",
            "check your understanding",
            "test yourself"
        ]
    ):

        return {
            "type": "assessment",
            "confidence": 0.90
        }



    # -----------------------------------
    # Practice Detection
    # -----------------------------------

    if any(
        word in text
        for word in [
            "practice",
            "exercise",
            "grounding",
            "tracking",
            "resourcing",
            "breathing",
            "meditation"
        ]
    ):

        return {
            "type": "practice",
            "confidence": 0.85
        }



    # -----------------------------------
    # Activity Detection
    # -----------------------------------

    if any(
        word in text
        for word in [
            "activity",
            "group discussion",
            "discussion"
        ]
    ):

        return {
            "type": "activity",
            "confidence": 0.80
        }
    if text.startswith(
      "imagine"
):

     return {
        "type":"example",
        "confidence":0.85
    }



    # -----------------------------------
    # Topic Detection
    # -----------------------------------

    if (
        is_bold
        and 11 <= font_size <= 18
        and 1 <= len(text.split()) <= 8
    ):

        return {
            "type": "topic",
            "confidence": 0.80
        }



    # -----------------------------------
    # General Content
    # -----------------------------------

    return {
        "type": "content",
        "confidence": 0.70
    }
    



