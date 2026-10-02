from pathlib import Path
import json
import re
from collections import Counter

from classifier.rule_classifier import classify
from knowledge.node_builder import build_node
from parser.metadata_extractor import extract_blocks
from parser.highlight_extractor import extract_highlighted_keywords
from knowledge.keyword_skill_mapper import map_keywords_to_skills
from knowledge.objective_mapper import map_nodes_to_objectives


BASE_DIR = Path(__file__).resolve().parents[1]

PDF_PATH = BASE_DIR / "data" / "SEEK_Learning.pdf"
OUTPUT_PATH = BASE_DIR / "output" / "knowledge_base.json"
REPORT_PATH = BASE_DIR / "output" / "validation_report.json"
HIGHLIGHTED_KEYWORDS_PATH = BASE_DIR / "output" / "highlighted_keywords.json"
KEYWORD_SKILL_MAP_PATH = BASE_DIR / "output" / "keyword_skill_map.json"
CONTENT_SKILLS_PATH = BASE_DIR / "output" / "content_skills.json"


def build_knowledge_base():

    print("Reading PDF...")

    if not PDF_PATH.exists():
        print("PDF not found:", PDF_PATH)
        return []


    # -------------------------------
    # 1. Extract PDF blocks
    # -------------------------------
    blocks = extract_blocks(PDF_PATH)

    print("Total extracted blocks:", len(blocks))


    # -------------------------------
    # 2. Extractor already returns paragraph/list-item blocks.
    # -------------------------------
    merged_blocks = blocks

    print("Merged blocks:", len(merged_blocks))


    nodes = []


    stack = {
        "chapter": None,
        "module": None,
        "topic": None,
        "practice": None,
        "activity": None,
        "reflection": None,
        "assessment": None,
    }

    objective_section = False


    # -------------------------------
    # 3. Create knowledge nodes
    # -------------------------------
    for index, block in enumerate(merged_blocks, start=1):

        # Clean extracted PDF text before classification and node creation.
        if "text" in block:
            block["text"] = clean_text(block["text"])

        if "title" in block:
            block["title"] = clean_text(block["title"])

        if "content" in block:
            block["content"] = clean_text(block["content"])

        title = block.get("title", "")
        content = block.get("content", "")
        text = block.get("text", "")

        if not title and not content and not text:
            continue

        if not is_valid_block(block):
            continue

        # Classification with confidence
        classification = classify(block)


        # Backward compatibility
        if isinstance(classification, dict):

            node_type = classification.get(
                "type",
                "content"
            )

            confidence = classification.get(
                "confidence",
                0.5
            )

        else:

            node_type = classification
            confidence = 0.5

        is_bold_heading = "bold" in block.get("font_name", "").lower()

        if node_type == "objective_heading":
            objective_section = True
            continue
        elif objective_section and (
            node_type in {"chapter", "topic"}
            or (
                is_bold_heading
                and not text.lower().startswith("by the end")
            )
        ):
            objective_section = False
        elif not objective_section and node_type == "learning_objective":
            node_type = "content"
            confidence = 0.70



        parent_id = resolve_parent_id(
            node_type,
            stack
        )


        node_id = f"node_{index:04d}"


        node = build_node(
            block,
            node_type,
            node_id,
            parent_id=parent_id
        )

        if node is None:
            continue


        # -------------------------------
        # Add enriched metadata
        # -------------------------------

        node["classification_confidence"] = confidence


        node["learning_objectives"] = generate_learning_objectives(
            node
        )


        node["prerequisites"] = []

        node["next_nodes"] = []


        node["tags"] = generate_tags(
            node
        )


        # Source citation
        node["source"] = {

            "document": "SEEK Learning",

            "page": block.get(
                "page",
                None
            )

        }


        node["validation_status"] = "pending"


        nodes.append(node)


        update_stack(
            stack,
            node_type,
            node_id
        )


    with open(CONTENT_SKILLS_PATH, "r", encoding="utf-8") as file:
        content_skills = json.load(file)

    nodes, _ = map_nodes_to_objectives(nodes, content_skills)
    return nodes



# --------------------------------------------------
# Parent-child hierarchy handling
# --------------------------------------------------

def resolve_parent_id(node_type, stack):

    if node_type == "chapter":
        return None


    if node_type == "module":
        return stack["chapter"]


    if node_type == "topic":
        return (
            stack["module"]
            or stack["chapter"]
        )


    if node_type in {
        "practice",
        "activity",
        "reflection",
        "assessment"
    }:

        return (
            stack["topic"]
            or stack["module"]
            or stack["chapter"]
        )


    return (
        stack["topic"]
        or stack["module"]
        or stack["chapter"]
    )



# --------------------------------------------------
# Update hierarchy stack
# --------------------------------------------------

def update_stack(stack, node_type, node_id):


    if node_type == "chapter":

        stack.update({

            "chapter": node_id,
            "module": None,
            "topic": None,
            "practice": None,
            "activity": None,
            "reflection": None,
            "assessment": None,

        })

        return



    if node_type == "module":

        stack.update({

            "module": node_id,
            "topic": None,
            "practice": None,
            "activity": None,
            "reflection": None,
            "assessment": None,

        })

        return



    if node_type == "topic":

        stack.update({

            "topic": node_id,
            "practice": None,
            "activity": None,
            "reflection": None,
            "assessment": None,

        })

        return



    if node_type in {

        "practice",
        "activity",
        "reflection",
        "assessment"

    }:

        stack[node_type] = node_id



# --------------------------------------------------
# Metadata generation helpers
# --------------------------------------------------

def generate_learning_objectives(node):
    """
    Do not generate artificial learning objectives.
    Only learning objectives explicitly extracted from the SEEK PDF
    should be stored as learning objective nodes.
    """

    if node.get("type") == "learning_objective":
        title = node.get("title", "").strip()
        content = node.get("content", "").strip()

        objective = content or title

        if objective:
            return [objective]

    return []

def clean_text(text):
    """
    Clean common PDF extraction artifacts.
    """

    if not text:
        return ""

    text = str(text)

    # Remove common PDF bullet symbols
    text = text.replace("\u2022", " ")
    text = text.replace("\u25cf", " ")
    text = text.replace("\uf0b7", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_valid_block(block):
    """Reject blocks that contain only symbols or meaningless text."""

    title = block.get("title", "")
    content = block.get("content", "")
    text = block.get("text", "")
    combined = f"{title} {content} {text}".strip()

    if not combined:
        return False

    meaningful = re.sub(r"[^\w\s]", "", combined, flags=re.UNICODE)
    return len(meaningful.strip()) >= 2


def generate_tags(node):

    title = node.get(
        "title",
        ""
    )


    words = title.lower().split()


    return words[:5]



# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_nodes(nodes, content_skills=None):

    report = {

        "total_nodes": len(nodes),

        "approved": 0,

        "requires_review": 0,

        "issues": [],
        "duplicate_ids": [],
        "duplicate_content": [],
        "invalid_objective_ids": [],
        "mapping_review": [],
        "possible_split_nodes": []

    }

    valid_objectives = {
        objective.get("objective_id")
        for skill in (content_skills or [])
        for objective in skill.get("objectives", [])
    }
    seen_ids = set()
    seen_content = {}

    for node in nodes:
        node_id = node.get("id")
        normalized_content = re.sub(r"\s+", " ", node.get("content", "").lower()).strip()
        if node_id in seen_ids:
            report["duplicate_ids"].append(node_id)
        seen_ids.add(node_id)
        if normalized_content:
            if node.get("type") not in {"topic", "reflection"}:
                seen_content.setdefault(normalized_content, []).append(node_id)

        objective_id = node.get("objective_id")
        if objective_id and objective_id not in valid_objectives:
            report["invalid_objective_ids"].append(node_id)
        if node.get("type") not in {"chapter", "module", "topic", "objective_heading"} and node.get("match_status") in {"review", "unassigned"}:
            report["mapping_review"].append({"node_id": node_id, "status": node.get("match_status")})

    report["duplicate_content"] = [
        node_ids for node_ids in seen_content.values() if len(node_ids) > 1
    ]

    for previous, current in zip(nodes, nodes[1:]):
        previous_text = previous.get("content", "").strip()
        current_text = current.get("content", "").strip()
        if (
            previous.get("parent_id") == current.get("parent_id")
            and previous.get("type") == current.get("type") == "content"
            and previous_text
            and current_text
            and previous_text[-1].isalnum()
            and current_text[0].islower()
        ):
            report["possible_split_nodes"].append([previous["id"], current["id"]])


    for node in nodes:


        issues = []


        if (
            not node.get("content")
            and node.get("type") not in {
                "learning_objective",
                "objective_heading"
            }
        ):

            issues.append(
                "Missing content"
            )


        if node.get(
            "classification_confidence",
            0
        ) < 0.6:

            issues.append(
                "Low classification confidence"
            )


        if issues:


            node["validation_status"] = "review"


            report["requires_review"] += 1


            report["issues"].append({

                "node_id": node["id"],

                "issues": issues

            })


        else:

            node["validation_status"] = "approved"

            report["approved"] += 1



    if report["duplicate_ids"] or report["duplicate_content"]:
        report["requires_review"] += len(report["duplicate_ids"]) + len(report["duplicate_content"])
    if report["invalid_objective_ids"]:
        report["requires_review"] += len(report["invalid_objective_ids"])
    report["issues"].extend(
        {"node_id": node_id, "issues": ["Invalid objective_id"]}
        for node_id in report["invalid_objective_ids"]
    )
    report["issues"].extend(
        {"node_id": pair[0], "issues": ["Possible paragraph split before node"]}
        for pair in report["possible_split_nodes"]
    )

    return nodes, report
from parser.highlight_extractor import extract_highlighted_keywords

KEYWORDS_OUTPUT_PATH = BASE_DIR / "output" / "highlighted_keywords.json"


# --------------------------------------------------
# Main execution
# --------------------------------------------------

if __name__ == "__main__":


    knowledge_base = build_knowledge_base()


    print(
        "\nRunning validation..."
    )


    with open(CONTENT_SKILLS_PATH, "r", encoding="utf-8") as file:
        content_skills = json.load(file)

    knowledge_base, report = validate_nodes(knowledge_base, content_skills)


    OUTPUT_PATH.parent.mkdir(
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:


        json.dump(
            knowledge_base,
            file,
            indent=4,
            ensure_ascii=False
        )


    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:


        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )


    print("\nCompleted!")

    print(
        "Total Nodes:",
        len(knowledge_base)
    )


    print(
        "Saved:",
        OUTPUT_PATH
    )


    print(
        "Validation Report:",
        REPORT_PATH
    )


    print(
        "\nNode Distribution:"
    )


    print(
        Counter(
            node["type"]
            for node in knowledge_base
        )
    )

    highlighted_keywords = extract_highlighted_keywords(PDF_PATH)

    with open(
        HIGHLIGHTED_KEYWORDS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            highlighted_keywords,
            file,
            indent=4,
            ensure_ascii=False
        )

    keyword_skill_map = map_keywords_to_skills(
        highlighted_keywords,
        knowledge_base
    )

    with open(
        KEYWORD_SKILL_MAP_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            keyword_skill_map,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Highlighted keywords:", len(highlighted_keywords))
    print("Saved:", HIGHLIGHTED_KEYWORDS_PATH)
    print("Saved:", KEYWORD_SKILL_MAP_PATH)

