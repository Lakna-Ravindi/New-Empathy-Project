"""Learning Recommendation Service for Empathy Guru.

Connects the General Empathy Chat with the approved UNESCO MGIEP SEEK curriculum
by analyzing completed conversation history, identifying emotional and interpersonal
themes, and matching them against the eight approved SEEK skills.

Constraints:
- Strictly recommends a maximum of 2 approved skills.
- Strictly validates every skill ID and objective ID against canonical curriculum data.
- Never invents skills, objectives, or learning content.
- Never diagnoses mental health conditions or treats recommendations as psychological evaluations.
- Employs strict confidence thresholds; returns zero recommendations when relevance is insufficient.
- Supports Gemini structured output if configured, with deterministic fallback for tests and offline operation.
- Prioritizes safety: immediately yields safety escalation when high-risk patterns are detected.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Base path resolution
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]

try:
    from knowledge.objective_mapper import CANONICAL_OBJECTIVE_IDS
except ImportError:
    # Standalone fallback if package structure varies
    CANONICAL_OBJECTIVE_IDS = {
        "skill_01": {1: "obj_01_01", 2: "obj_01_05", 3: "obj_01_02", 4: "obj_01_06", 5: "obj_01_03", 6: "obj_01_04"},
        "skill_02": {1: "obj_02_01", 2: "obj_02_02", 3: "obj_02_03", 4: "obj_02_04", 5: "obj_02_05"},
        "skill_03": {1: "obj_03_01", 2: "obj_03_05", 3: "obj_03_06", 4: "obj_03_02", 5: "obj_03_03", 6: "obj_03_04"},
        "skill_04": {1: "obj_04_01", 2: "obj_04_02", 3: "obj_04_05", 4: "obj_04_06", 5: "obj_04_07", 6: "obj_04_03", 7: "obj_04_04"},
        "skill_05": {1: "obj_05_03", 2: "obj_05_01", 3: "obj_05_02", 4: "obj_05_04", 5: "obj_05_05"},
        "skill_06": {1: "obj_06_01", 2: "obj_06_02", 3: "obj_06_05", 4: "obj_06_03", 5: "obj_06_06", 6: "obj_06_04"},
        "skill_07": {1: "obj_07_01", 2: "obj_07_02", 3: "obj_07_03", 4: "obj_07_05", 5: "obj_07_06", 6: "obj_07_04"},
        "skill_08": {1: "obj_08_01", 2: "obj_08_05", 3: "obj_08_02", 4: "obj_08_03", 5: "obj_08_04"},
    }

try:
    from safety import detect_safety_risk
except ImportError:
    try:
        from backend.safety import detect_safety_risk
    except ImportError:
        def detect_safety_risk(text: str) -> Dict[str, Any]:
            return {"qualifies": False, "risk_level": "none", "immediate": False}

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


@dataclass
class RecommendationItem:
    """Structured contract for a single SEEK skill recommendation."""
    skill_id: str
    skill_title: str
    objective_id: str
    objective_title: str
    relevance_explanation: str
    learning_opportunity: str
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_title": self.skill_title,
            "objective_id": self.objective_id,
            "objective_title": self.objective_title,
            "relevance_explanation": self.relevance_explanation,
            "learning_opportunity": self.learning_opportunity,
            "confidence_score": round(self.confidence_score, 3),
        }


# ==============================================================================
# THEMATIC PATTERNS & HEURISTICS FOR ALL 8 APPROVED SEEK SKILLS
# ==============================================================================

SKILL_THEMATIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "skill_01": {
        "title": "Calming the Body and Mind",
        "primary_objective": "obj_01_04",
        "theme_patterns": [
            r"\b(body|physic(al|ally)|sensation|somatic|nervous system|autonomic)\b",
            r"\b(breath(e|ing)?|shallow breath|hyperventilat(e|ing)|suffocat(e|ing)?)\b",
            r"\b(heart rac(e|ing)|rapid heart|chest tight(ness)?|palpitation)\b",
            r"\b(shak(e|ing)|trembl(e|ing)|jitter(y)?|sweat(y|ing)?|chills)\b",
            r"\b(tension|tense muscles?|tightness in (my|the)? ?(shoulders|neck|jaw|chest))\b",
            r"\b(ground(ing)?|track(ing)?|resourc(e|ing)?|resilient zone)\b",
            r"\b(panick?(y|ed)?|anxious in my body|numb(ness)?|frozen|dizz(y|iness))\b",
            r"\b(stomach knot|knot in my stomach|butterflies|nausea(ted)?)\b",
        ],
        "objective_matchers": {
            "obj_01_01": [r"\b(name|identif(y|ying)|label(ing)?|describ(e|ing))\b.*\b(sensation|feeling in (my|the) body)\b", r"\bbody literac(y)?\b"],
            "obj_01_02": [r"\b(resilient zone|hyper-?arousal|hypo-?arousal|dysregulat(ed|ion)|out of zone)\b"],
            "obj_01_03": [r"\b(sympathetic|parasympathetic|fight or flight|rest and digest)\b"],
            "obj_01_04": [r"\b(ground(ing)?|track(ing)?|resourc(e|ing)?|calm(ing)? (down|myself)|relax(ing|ation)?)\b", r"\bstress\b"],
            "obj_01_05": [r"\b(vocabulary|words for|term for)\b.*\b(body|sensation)\b"],
            "obj_01_06": [r"\b(out of (my|the) (resilience|resilient) zone|overwhelmed physically)\b"],
        },
        "explanation_templates": [
            "In your conversation, you mentioned physical tension and bodily stress. Calming the Body and Mind teaches grounding and somatic tracking to help you recognize physical cues and restore balance.",
            "You touched on feeling physically unsettled or overwhelmed. This skill offers practical tools to understand your body's stress signals and return to your resilient zone.",
        ],
    },
    "skill_02": {
        "title": "Ethical Mindfulness",
        "primary_objective": "obj_02_01",
        "theme_patterns": [
            r"\b(mindful(ness)?|mind-wandering|wandering mind|distract(ed|ion)?)\b",
            r"\b(ethic(al|s)?|moral|values?|value-aligned|integrity|conscience)\b",
            r"\b(heedful(ness)?|awareness|attent(ion|ive)|focus(ed|ing)?)\b",
            r"\b(impuls(ive|ively|ivity)|rash decision|regret(ted)? doing|acted without thinking)\b",
            r"\b(dilemma|right thing to do|ethical choice|moral dilemma)\b",
            r"\b(metacognit(ion|ive)|observ(e|ing) thoughts)\b",
        ],
        "objective_matchers": {
            "obj_02_01": [r"\b(ethic(al)? mindfulness|decision-?making under pressure|pressure|value-aligned decision)\b"],
            "obj_02_02": [r"\b(heedfulness|mindfulness and awareness|three core components)\b"],
            "obj_02_03": [r"\b(metacognitive|stabiliz(e|ing) attention|prevent(ing)? cognitive wandering)\b"],
            "obj_02_04": [r"\b(behavioral choices?|personal and professional|actions aligned with values)\b"],
            "obj_02_05": [r"\b(influence outcomes|scenario analysis)\b"],
        },
        "explanation_templates": [
            "You spoke about navigating choices and staying attentive under pressure. Ethical Mindfulness helps anchor your focus and align your actions with what genuinely matters to you.",
            "In your chat, you reflected on thoughtful decision-making and awareness. This skill supports developing steady attention and responding deliberately rather than reactively.",
        ],
    },
    "skill_03": {
        "title": "Emotional Awareness",
        "primary_objective": "obj_03_02",
        "theme_patterns": [
            r"\b(spark to forest fire|escalat(e|ing|ion)|snowball(ed|ing)?)\b",
            r"\b(trigger(s|ed|ing)?|preconditions?|irritat(ed|ion)|annoy(ed|ing)?)\b",
            r"\b(lost my temper|temper flared|snapped at|outburst|explos(ive|ion))\b",
            r"\b(rage|furious|boiling with anger|anger took over)\b",
            r"\b(selective filtering|pause before react(ing)?|cognitive space)\b",
            r"\b(beneficial mental states?|harmful mental states?|afflictive)\b",
            r"\b(emotional(ly)? aware(ness)?|recogniz(e|ing) (my|the) emotions?)\b",
        ],
        "objective_matchers": {
            "obj_03_01": [r"\b(beneficial and harmful|beneficial mental states|harmful mental states)\b"],
            "obj_03_02": [r"\b(spark to forest fire|stages of emotional escalation|escalat(e|ion))\b"],
            "obj_03_03": [r"\b(selective filtering|cognitive space|between trigger and action)\b"],
            "obj_03_04": [r"\b(personal triggers?|preconditions|emotional stability)\b"],
            "obj_03_05": [r"\b(importance of emotional awareness)\b"],
            "obj_03_06": [r"\b(afflictive|harmful vs beneficial)\b"],
        },
        "explanation_templates": [
            "You shared how quickly emotional reactions and irritation can build up. Emotional Awareness introduces the 'Spark to Forest Fire' model to recognize triggers early before reactions escalate.",
            "In your conversation, you reflected on intense emotional moments. This skill helps build cognitive space between an emotional spark and your response.",
        ],
    },
    "skill_04": {
        "title": "Self-Compassion",
        "primary_objective": "obj_04_04",
        "theme_patterns": [
            r"\b(self-critic(al|ism)?|harsh on myself|hard on myself|beat(ing)? myself up)\b",
            r"\b(not good enough|feel like a failure|failure|failed|inadequat(e|y))\b",
            r"\b(perfectionis(t|m)?|unrealistic expectations?|flawless|must be perfect)\b",
            r"\b(hate myself|blam(e|ing) myself|my fault entirely|shame|ashamed)\b",
            r"\b(hedonic treadmill|eight worldly concerns|superiority bias|inferiority bias)\b",
            r"\b(self-worth|intrinsic worth|self-compassion|kind to myself)\b",
            r"\b(impost(or|er)( syndrome)?|compare myself to (others|everyone))\b",
        ],
        "objective_matchers": {
            "obj_04_01": [r"\b(intrinsic self-worth|self-worth|independent of external conditions)\b"],
            "obj_04_02": [r"\b(cognitive distortions?|hedonic treadmill|perfectionism|self-biases)\b"],
            "obj_04_03": [r"\b(eight worldly concerns|praise and blame|gain and loss)\b"],
            "obj_04_04": [r"\b(replace harsh self-criticism|constructive encouragement|harsh|critic(al|ism))\b"],
            "obj_04_05": [r"\b(importance of self-compassion)\b"],
            "obj_04_06": [r"\b(examine if you practice self-compassion)\b"],
            "obj_04_07": [r"\b(three steps to cultivate compassion)\b"],
        },
        "explanation_templates": [
            "You reflected on feeling critical of yourself or holding impossible expectations. Self-Compassion teaches how to replace harsh inner criticism with constructive encouragement and recognize your self-worth.",
            "In your chat, you expressed feeling inadequate or doubting yourself. This skill provides structured tools to navigate perfectionism and cultivate kindness toward yourself.",
        ],
    },
    "skill_05": {
        "title": "Impartiality and Common Humanity",
        "primary_objective": "obj_05_01",
        "theme_patterns": [
            r"\b(common humanity|shared desire|we all (suffer|want happiness))\b",
            r"\b(us vs them|in-group|out-group|outsider|alienat(ed|ion))\b",
            r"\b(prejudice|stereotyp(e|es|ing)?|bias(ed)?|categoric(al)? bias)\b",
            r"\b(impartial(ity)?|treating everyone identically|essentialism)\b",
            r"\b(perspective-taking|see(ing)? their point of view|in their shoes)\b",
            r"\b(feel(ing)? isolated|no one understands|separated from everyone)\b",
        ],
        "objective_matchers": {
            "obj_05_01": [r"\b(common humanity|seek well-being and avoid suffering|shared desire)\b"],
            "obj_05_02": [r"\b(differentiate between impartiality and treating everyone identically)\b"],
            "obj_05_03": [r"\b(in-group bias|stereotypes|prejudice|origins and consequences)\b"],
            "obj_05_04": [r"\b(perspective-taking|weaken implicit categorical bias)\b"],
            "obj_05_05": [r"\b(practice tools to cultivate impartiality)\b"],
        },
        "explanation_templates": [
            "You touched on feeling separate from others or noticing divisions and biases. Impartiality and Common Humanity explores our shared desire for well-being and breaks down divisive assumptions.",
            "In your conversation, perspective-taking and connection emerged as meaningful themes. This skill helps expand your sense of belonging and bridge interpersonal divides.",
        ],
    },
    "skill_06": {
        "title": "Forgiveness and Gratitude",
        "primary_objective": "obj_06_01",
        "theme_patterns": [
            r"\b(forgiv(e|ing|eness)?|unforgiving|can(')?t let go|resent(ment|ful)?)\b",
            r"\b(grudge|bitter(ness)?|holding onto anger|betray(ed|al)?|hurt by (him|her|them|friend))\b",
            r"\b(fatal attribution error|reductive thinking|causes and conditions)\b",
            r"\b(gratitude|grateful|thankful(ness)?|appreciat(e|ion|ive))\b",
            r"\b(naikan( practice)?|interdependen(ce|t)|acknowledge support)\b",
            r"\b(reconcil(e|iation)?|letting go of resentment)\b",
        ],
        "objective_matchers": {
            "obj_06_01": [r"\b(define forgiveness|letting go of resentment|without condoning)\b"],
            "obj_06_02": [r"\b(contextual analysis|causes and conditions|fatal attribution error|reductive thinking)\b"],
            "obj_06_03": [r"\b(state gratitude|trait gratitude|distinguish between state)\b"],
            "obj_06_04": [r"\b(naikan|japanese naikan|three-question|interdependence)\b"],
            "obj_06_05": [r"\b(thinking traps to forgiveness)\b"],
            "obj_06_06": [r"\b(strategies to cultivate gratitude)\b"],
        },
        "explanation_templates": [
            "You spoke about lingering hurt, resentment, or working through interpersonal grievances. Forgiveness and Gratitude offers ways to look at causes and conditions so you can release resentment at your own pace.",
            "In your chat, letting go of resentment and recognizing appreciation were central themes. This skill provides gentle, reflective exercises like Naikan to foster emotional peace.",
        ],
    },
    "skill_07": {
        "title": "Empathic Concern",
        "primary_objective": "obj_07_06",
        "theme_patterns": [
            r"\b(empath(y|ic|etic)?|empathic concern|affective empathy|cognitive empathy)\b",
            r"\b(empathic distress|empathy fatigue|compassion fatigue|burnout from care|drained from helping)\b",
            r"\b(emotional contagion|absorbing (their|other people(')?s) (pain|feelings|emotions?))\b",
            r"\b(taking on too much|feeling what they feel too much|burdened by others(')? problems)\b",
            r"\b(sympathetic joy|jealous of (friend|others?)|envy|celebrat(e|ing) others)\b",
            r"\b(one-over-many trap)\b",
        ],
        "objective_matchers": {
            "obj_07_01": [r"\b(affective empathy from cognitive empathy|define empathy)\b"],
            "obj_07_02": [r"\b(emotional contagion|evolutionary mechanism)\b"],
            "obj_07_03": [r"\b(cognitive traps|one-over-many trap|in-group bias in empathy)\b"],
            "obj_07_04": [r"\b(sympathetic joy|counter jealousy|celebrate the success of others)\b"],
            "obj_07_05": [r"\b(significance of empathic concern)\b"],
            "obj_07_06": [r"\b(empathic distress|strategies to reduce it|burnout|drained)\b"],
        },
        "explanation_templates": [
            "You talked about feeling emotionally drained or absorbing the weight of others' struggles. Empathic Concern explores ways to balance care with healthy boundaries so you avoid empathic distress.",
            "In your conversation, navigating strong feelings around others' emotions and successes was evident. This skill helps distinguish between taking on others' pain and sustaining caring empathy.",
        ],
    },
    "skill_08": {
        "title": "Compassion",
        "primary_objective": "obj_08_01",
        "theme_patterns": [
            r"\b(compassion(ate)?|noticing suffering|alleviat(e|ing) suffering|reliev(e|ing) suffering)\b",
            r"\b(sense of agency|helpless(ness)? to help|want to help but don't know how)\b",
            r"\b(compassion is (not a)? ?weakness|passiv(e|ity)|courageous motivation)\b",
            r"\b(wishing compassion|aspiring compassion|engaged compassion)\b",
            r"\b(action-oriented care|step up for someone|support someone suffering)\b",
        ],
        "objective_matchers": {
            "obj_08_01": [r"\b(courageous motivation|relieve suffering|critical inquiry|define compassion)\b"],
            "obj_08_02": [r"\b(misconceptions?|not a weakness|passivity)\b"],
            "obj_08_03": [r"\b(three essential components|noticing suffering|sense of agency)\b"],
            "obj_08_04": [r"\b(wishing|aspiring|engaged levels|personal and social)\b"],
            "obj_08_05": [r"\b(thinking traps to compassion)\b"],
        },
        "explanation_templates": [
            "You expressed a desire to support others facing hardships while feeling unsure how to make a meaningful difference. Compassion focuses on translating caring concern into grounded, active agency.",
            "In your chat, you reflected on helping others through difficulty. This skill helps build courageous, active compassion while clarifying that compassion is a source of strength, not passivity.",
        ],
    },
}


class LearningRecommendationService:
    """Production service generating context-aware SEEK learning recommendations.

    Matches conversation history to the 8 approved UNESCO MGIEP SEEK curriculum skills
    using canonical objective mappings, structured LLM reasoning (when configured),
    and a reliable deterministic rule-based fallback engine.
    """

    def __init__(
        self,
        content_skills_path: Optional[Union[str, Path]] = None,
        progress_structure_path: Optional[Union[str, Path]] = None,
        client: Optional[Any] = None,
        confidence_threshold: float = 0.35,
        max_recommendations: int = 2,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_recommendations = max(1, min(max_recommendations, 2))
        self._client = client

        # Canonical data caches
        self.skills_by_id: Dict[str, Dict[str, Any]] = {}
        self.objectives_by_skill: Dict[str, Dict[str, str]] = {}
        self.canonical_skills: Set[str] = set(CANONICAL_OBJECTIVE_IDS.keys())
        self.canonical_objectives: Set[str] = {
            obj_id
            for objs in CANONICAL_OBJECTIVE_IDS.values()
            for obj_id in objs.values()
        }

        self._load_curriculum(content_skills_path, progress_structure_path)

    def _load_curriculum(
        self,
        content_skills_path: Optional[Union[str, Path]],
        progress_structure_path: Optional[Union[str, Path]],
    ) -> None:
        """Loads canonical content_skills.json and progress_structure.json."""
        candidate_content_paths = [
            Path(content_skills_path) if content_skills_path else None,
            ROOT_DIR / "output" / "content_skills.json",
            BACKEND_DIR.parent / "output" / "content_skills.json",
            Path(os.getenv("CONTENT_KB_PATH", "")) if os.getenv("CONTENT_KB_PATH") else None,
        ]

        loaded_content = None
        for p in candidate_content_paths:
            if p and p.is_file():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        loaded_content = json.load(f)
                    break
                except Exception as e:
                    logger.warning("Could not load content skills from %s: %s", p, e)

        if isinstance(loaded_content, list):
            for skill in loaded_content:
                sid = skill.get("skill_id")
                if sid in self.canonical_skills:
                    self.skills_by_id[sid] = {
                        "skill_id": sid,
                        "title": skill.get("skill_title") or skill.get("title", ""),
                        "description": skill.get("description", ""),
                        "key_concepts": skill.get("key_concepts", []),
                    }
                    objs = {}
                    for obj in skill.get("objectives", []):
                        oid = obj.get("objective_id") or obj.get("id")
                        if oid in self.canonical_objectives:
                            objs[oid] = obj.get("objective_title") or obj.get("title", "")
                    self.objectives_by_skill[sid] = objs

        # Fallback to profiles if files not on disk
        for sid, profile in SKILL_THEMATIC_PROFILES.items():
            if sid not in self.skills_by_id:
                self.skills_by_id[sid] = {
                    "skill_id": sid,
                    "title": profile["title"],
                    "description": f"Approved UNESCO MGIEP curriculum skill for {profile['title']}.",
                    "key_concepts": [],
                }
            if sid not in self.objectives_by_skill:
                self.objectives_by_skill[sid] = {
                    obj_id: f"Core objective {obj_id} for {profile['title']}"
                    for obj_id in CANONICAL_OBJECTIVE_IDS.get(sid, {}).values()
                }

    def recommend(
        self,
        conversation_history: List[Dict[str, Any]],
        student_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyzes General Empathy conversation history and recommends relevant SEEK skills.

        Args:
            conversation_history: List of conversation turn dicts with 'role' and 'content'.
            student_id: Optional student identifier.

        Returns:
            Structured dictionary with:
              - status: 'success', 'no_recommendations', or 'safety_escalation'
              - recommendations: list of recommendation dicts (max 2)
        """
        # 1. Validation & sanitization of conversation history
        if not isinstance(conversation_history, list) or not conversation_history:
            return {"status": "no_recommendations", "recommendations": []}

        sanitized_texts: List[str] = []
        user_texts: List[str] = []

        for turn in conversation_history:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                clean_text = content.strip()
                sanitized_texts.append(clean_text)
                if role == "user":
                    user_texts.append(clean_text)

        if not user_texts:
            return {"status": "no_recommendations", "recommendations": []}

        # 2. Safety pre-check: ensure no high-risk or crisis turns bypass safety escalation
        for text in user_texts:
            risk = detect_safety_risk(text)
            if risk.get("qualifies"):
                logger.warning("Safety risk detected in conversation context. Halting learning recommendations.")
                return {
                    "status": "safety_escalation",
                    "safety": True,
                    "recommendations": [],
                }

        combined_text = " ".join(user_texts)
        full_context_text = " ".join(sanitized_texts)

        # 3. If Gemini is available, attempt structured LLM reasoning
        recommendations = self._try_gemini_recommend(combined_text, full_context_text)

        # 4. If Gemini is unavailable, errors, or returns unvalidated recommendations, use deterministic engine
        if recommendations is None:
            recommendations = self._deterministic_recommend(combined_text)

        # 5. Strict canonical validation & maximum capping
        validated_recommendations = self._validate_and_sanitize_recommendations(recommendations)

        if not validated_recommendations:
            return {"status": "no_recommendations", "recommendations": []}

        return {
            "status": "success",
            "recommendations": [r.to_dict() for r in validated_recommendations[: self.max_recommendations]],
        }

    def _deterministic_recommend(self, text: str) -> List[RecommendationItem]:
        """Deterministic matching engine evaluating conversation themes against all 8 skills."""
        cleaned_text = text.lower()

        # Stopword / trivial conversation check
        words = re.findall(r"\b[a-z]{2,}\b", cleaned_text)
        trivial_greetings = {
            "hi", "hello", "hey", "good", "morning", "afternoon", "evening",
            "thank", "thanks", "ok", "okay", "bye", "yes", "no", "sure", "fine",
        }
        meaningful_words = [w for w in words if w not in trivial_greetings]
        if len(meaningful_words) < 2:
            return []

        scored_skills: List[Tuple[str, float, str]] = []

        for skill_id, profile in SKILL_THEMATIC_PROFILES.items():
            score = 0.0
            matched_themes: List[str] = []

            for pattern in profile["theme_patterns"]:
                matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
                if matches:
                    score += 0.25 * len(matches)
                    matched_themes.append(pattern)

            # Check key concepts from canonical skill
            skill_info = self.skills_by_id.get(skill_id, {})
            for concept in skill_info.get("key_concepts", []):
                concept_lower = concept.lower()
                if concept_lower in cleaned_text:
                    score += 0.35

            if score > 0:
                # Normalize score between 0.0 and 1.0
                normalized_score = min(1.0, score / 1.5)
                if normalized_score >= self.confidence_threshold:
                    scored_skills.append((skill_id, normalized_score, " ".join(matched_themes)))

        # Sort by score descending
        scored_skills.sort(key=lambda x: -x[1])

        results: List[RecommendationItem] = []
        for skill_id, score, match_info in scored_skills[: self.max_recommendations]:
            profile = SKILL_THEMATIC_PROFILES[skill_id]
            skill_info = self.skills_by_id.get(skill_id, {})
            skill_title = skill_info.get("title") or profile["title"]
            learning_opportunity = skill_info.get("description") or f"Learn core skills in {skill_title}."

            # Select best objective
            objective_id = self._select_best_objective(skill_id, cleaned_text)
            objective_title = self.objectives_by_skill.get(skill_id, {}).get(
                objective_id,
                f"Objective for {skill_title}",
            )

            # Generate context explanation
            explanation = self._generate_explanation(skill_id, profile)

            results.append(
                RecommendationItem(
                    skill_id=skill_id,
                    skill_title=skill_title,
                    objective_id=objective_id,
                    objective_title=objective_title,
                    relevance_explanation=explanation,
                    learning_opportunity=learning_opportunity,
                    confidence_score=score,
                )
            )

        return results

    def _select_best_objective(self, skill_id: str, text: str) -> str:
        """Chooses the most contextually relevant approved objective for a given skill."""
        profile = SKILL_THEMATIC_PROFILES.get(skill_id, {})
        objective_matchers = profile.get("objective_matchers", {})

        for obj_id, patterns in objective_matchers.items():
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    if obj_id in self.canonical_objectives:
                        return obj_id

        # Return primary fallback objective from profile
        primary = profile.get("primary_objective")
        if primary and primary in self.canonical_objectives:
            return primary

        # Fallback to first canonical objective registered for this skill
        valid_objs = CANONICAL_OBJECTIVE_IDS.get(skill_id, {})
        if valid_objs:
            return next(iter(valid_objs.values()))

        return "obj_01_01"

    def _generate_explanation(self, skill_id: str, profile: Dict[str, Any]) -> str:
        """Selects a warm, non-diagnostic relevance explanation."""
        templates = profile.get("explanation_templates", [])
        if templates:
            return templates[0]
        return f"This conversation touched on themes related to {profile.get('title')}. Exploring this skill offers practical self-reflection opportunities."

    def _try_gemini_recommend(
        self, user_text: str, full_context_text: str
    ) -> Optional[List[RecommendationItem]]:
        """Invokes Gemini LLM if configured; returns None on unavailability or failure."""
        client = self._client
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

        if client is None and api_key and genai is not None:
            try:
                client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.warning("Could not initialize Gemini client: %s", e)
                return None

        if client is None:
            return None

        prompt = f"""
You are the Smart Learning Bridge recommendation engine for Empathy Guru.
A student just completed a supportive General Empathy conversation.
Your task is to review the conversation and determine whether up to two of the eight approved SEEK curriculum skills are genuinely relevant.

THE EIGHT APPROVED SEEK SKILLS (ONLY SELECT FROM THESE):
1. skill_01: Calming the Body and Mind (Body literacy, somatic tracking, nervous system, resilient zone)
2. skill_02: Ethical Mindfulness (Heedfulness, awareness, value-aligned choices, attention stability)
3. skill_03: Emotional Awareness (Spark to forest fire, emotional escalation, identifying triggers)
4. skill_04: Self-Compassion (Navigating harsh self-criticism, perfectionism, intrinsic self-worth)
5. skill_05: Impartiality and Common Humanity (Common humanity, reducing bias, perspective-taking)
6. skill_06: Forgiveness and Gratitude (Letting go of resentment, Naikan reflection, gratitude)
7. skill_07: Empathic Concern (Empathic distress, emotional contagion, sympathetic joy)
8. skill_08: Compassion (Active caring, sense of agency, courageous compassion)

CONVERSATION CONTEXT:
{user_text}

INSTRUCTIONS:
- Recommend 0, 1, or at most 2 skills that genuinely fit the themes.
- If the conversation was casual, irrelevant, or lacks emotional/interpersonal depth, recommend 0 skills.
- Do NOT diagnose any mental health conditions (no depression, anxiety disorders, PTSD).
- Write a short, empathetic explanation (25-45 words) explaining why this skill is relevant.
- Return ONLY valid JSON matching this schema:
{{
  "recommendations": [
    {{
      "skill_id": "skill_01",
      "relevance_explanation": "explanation...",
      "confidence_score": 0.85
    }}
  ]
}}
"""
        try:
            config = None
            if types and hasattr(types, "GenerateContentConfig"):
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )

            gen_kwargs = {"model": model_name, "contents": prompt}
            if config:
                gen_kwargs["config"] = config

            result = client.models.generate_content(**gen_kwargs)
            if not result or not getattr(result, "text", None):
                return None

            cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(result.text).strip(), flags=re.I)
            payload = json.loads(cleaned_text)
            recs_data = payload.get("recommendations", [])

            if not isinstance(recs_data, list):
                return None

            gemini_results: List[RecommendationItem] = []
            for item in recs_data[: self.max_recommendations]:
                sid = item.get("skill_id")
                score = float(item.get("confidence_score", 0.8))
                if sid not in self.canonical_skills or score < self.confidence_threshold:
                    continue

                profile = SKILL_THEMATIC_PROFILES.get(sid, {})
                skill_info = self.skills_by_id.get(sid, {})
                skill_title = skill_info.get("title") or profile.get("title", sid)
                learning_opportunity = skill_info.get("description") or f"Learn core skills in {skill_title}."
                obj_id = self._select_best_objective(sid, user_text)
                obj_title = self.objectives_by_skill.get(sid, {}).get(obj_id, f"Objective for {skill_title}")

                explanation = str(item.get("relevance_explanation") or "").strip()
                if not explanation:
                    explanation = self._generate_explanation(sid, profile)

                gemini_results.append(
                    RecommendationItem(
                        skill_id=sid,
                        skill_title=skill_title,
                        objective_id=obj_id,
                        objective_title=obj_title,
                        relevance_explanation=explanation,
                        learning_opportunity=learning_opportunity,
                        confidence_score=score,
                    )
                )

            return gemini_results

        except Exception as err:
            logger.warning("Gemini recommendation call failed; falling back to deterministic engine: %s", err)
            return None

    def _validate_and_sanitize_recommendations(
        self, recommendations: Sequence[RecommendationItem]
    ) -> List[RecommendationItem]:
        """Validates all recommendations against canonical data and enforces constraints."""
        seen_skills: Set[str] = set()
        validated: List[RecommendationItem] = []

        for item in recommendations:
            if not isinstance(item, RecommendationItem):
                continue
            # 1. Validate skill_id
            if item.skill_id not in self.canonical_skills:
                logger.warning("Rejected non-canonical skill_id: %s", item.skill_id)
                continue
            if item.skill_id in seen_skills:
                continue

            # 2. Validate objective_id against the canonical objectives for this skill
            valid_skill_objs = CANONICAL_OBJECTIVE_IDS.get(item.skill_id, {})
            canonical_obj_list = set(valid_skill_objs.values())
            if item.objective_id not in canonical_obj_list:
                # If objective ID is invalid, fallback to the canonical primary objective
                primary = SKILL_THEMATIC_PROFILES.get(item.skill_id, {}).get("primary_objective")
                if primary and primary in canonical_obj_list:
                    item.objective_id = primary
                else:
                    item.objective_id = next(iter(canonical_obj_list))

            # 3. Synchronize canonical titles from verified records
            skill_info = self.skills_by_id.get(item.skill_id, {})
            if skill_info.get("title"):
                item.skill_title = skill_info["title"]
            if skill_info.get("description"):
                item.learning_opportunity = skill_info["description"]

            registered_obj_title = self.objectives_by_skill.get(item.skill_id, {}).get(item.objective_id)
            if registered_obj_title:
                item.objective_title = registered_obj_title

            # 4. Check confidence
            if item.confidence_score < self.confidence_threshold:
                continue

            seen_skills.add(item.skill_id)
            validated.append(item)

            if len(validated) >= self.max_recommendations:
                break

        return validated
