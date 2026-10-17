from pathlib import Path
import json
import re
from collections import Counter

from classifier.rule_classifier import classify
from knowledge.node_builder import build_node
from parser.metadata_extractor import extract_blocks
from parser.highlight_extractor import (
    extract_highlighted_keywords,
)
from knowledge.objective_mapper import map_nodes_to_objectives
from knowledge.progress_builder import build_content_progress_structure


BASE_DIR = Path(__file__).resolve().parents[1]

PDF_PATH = BASE_DIR / "data" / "SEEK_Learning.pdf"
OUTPUT_PATH = BASE_DIR / "output" / "knowledge_base.json"
REPORT_PATH = BASE_DIR / "output" / "validation_report.json"
HIGHLIGHTED_KEYWORDS_PATH = BASE_DIR / "output" / "highlighted_keywords.json"
CONTENT_SKILLS_PATH = BASE_DIR / "output" / "content_skills.json"
PROGRESS_STRUCTURE_PATH = BASE_DIR / "output" / "progress_structure.json"


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
    merged_blocks = split_combined_objective_blocks(blocks)

    print("Merged blocks:", len(merged_blocks))


    nodes = []


    stack = {
        "chapter": None,
        "module": None,
        "topic": None,
        "objective": None,
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

        explicit_objective = (
            isinstance(classification, dict)
            and classification.get("explicit", False)
        )
        is_bold_heading = "bold" in block.get("font_name", "").lower()

        if node_type == "objective_heading":
            objective_section = True
        elif objective_section and (
            node_type in {"chapter", "topic"}
            or (
                is_bold_heading
                and not text.lower().startswith("by the end")
            )
        ):
            objective_section = False
        elif (
            not objective_section
            and node_type == "learning_objective"
            and not explicit_objective
        ):
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
            ),
            "block_index": block.get("block_index"),
            "item_index": block.get("item_index"),
            "text": block.get("text", ""),

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


def split_combined_objective_blocks(blocks):
    """Split only numbered lists found in an explicit objective section."""
    result = []
    in_objective_section = False

    for block in blocks:
        text = block.get("text", "").strip()
        normalized = re.sub(r"\s+", " ", text).lower()
        if normalized in {
            "module objectives",
            "module learning objectives",
            "learning objectives",
            "objectives",
            "module objective",
            "module overview & objectives",
        }:
            in_objective_section = True
            result.append(block)
            continue

        if normalized.startswith("by the end of this skill") or normalized.startswith(
            "by the end of this module"
        ) or normalized.startswith("you will be able to"):
            in_objective_section = True
            result.append(block)
            continue

        if re.match(r"^skill\s+\d+\s*:", normalized):
            in_objective_section = False

        matches = list(re.finditer(r"(?<!\w)(\d+)[.)]\u200b*\s+", text))
        if in_objective_section and len(matches) > 1 and matches[0].start() == 0:
            for match_index, match in enumerate(matches):
                start = match.start()
                end = (
                    matches[match_index + 1].start()
                    if match_index + 1 < len(matches)
                    else len(text)
                )
                item = dict(block)
                item["text"] = text[start:end].strip()
                item["item_index"] = match_index
                result.append(item)
            continue

        result.append(block)

    return result



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
            or stack["objective"]
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
            or stack["objective"]
            or stack["module"]
            or stack["chapter"]
        )

    if node_type == "learning_objective":
        return (
            stack["topic"]
            or stack["module"]
            or stack["chapter"]
        )

    return (
        stack["topic"]
        or stack["objective"]
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
            "objective": None,
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
            "objective": None,
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


    if node_type == "learning_objective":
        stack["objective"] = node_id
        stack["topic"] = None
        stack["practice"] = None
        stack["activity"] = None
        stack["reflection"] = None
        stack["assessment"] = None
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

    text = text.replace("\u200b", "").replace("\ufeff", "")

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

def validate_nodes(nodes, content_skills=None, source_blocks=None):

    report = {

        "total_nodes": len(nodes),

        "approved": 0,

        "requires_review": 0,

        "issues": [],
        "duplicate_ids": [],
        "duplicate_content": [],
        "invalid_objective_ids": [],
        "mapping_review": [],
        "objective_mapping_review": [],
        "content_mapping_review": [],
        "objective_coverage": {},
        "intentional_duplicate_content": [],
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
            if node.get("type") not in {"topic", "reflection", "objective_heading"}:
                seen_content.setdefault(normalized_content, []).append(node_id)
            elif node.get("type") == "objective_heading":
                report["intentional_duplicate_content"].append(node_id)

        objective_id = node.get("objective_id")
        if objective_id and objective_id not in valid_objectives:
            report["invalid_objective_ids"].append(node_id)
        if (
            node.get("type") not in {"chapter", "module", "topic", "objective_heading"}
            and node.get("match_status") in {"review", "unassigned"}
        ):
            mapping = {"node_id": node_id, "status": node.get("match_status")}
            report["mapping_review"].append(mapping)
            if node.get("type") == "learning_objective":
                report["objective_mapping_review"].append(mapping)
            elif node.get("type") == "content":
                report["content_mapping_review"].append(mapping)

    report["duplicate_content"] = [
        node_ids for node_ids in seen_content.values() if len(node_ids) > 1
    ]

    objective_nodes = [
        node for node in nodes
        if node.get("type") == "learning_objective"
    ]
    source_objectives = [
        block.get("text", "").strip()
        for block in (source_blocks or [])
        if re.match(r"^objective\s+\d+\s*:", block.get("text", "").strip(), re.IGNORECASE)
    ]
    objective_titles = {
        re.sub(r"\s+", " ", node.get("title", "")).strip()
        for node in objective_nodes
    }
    missing_objectives = [
        title for title in source_objectives
        if re.sub(r"\s+", " ", title).strip() not in objective_titles
    ]
    report["objective_coverage"] = {
        "source_explicit_objective_count": len(source_objectives),
        "generated_objective_node_count": len(objective_nodes),
        "missing_source_objectives": missing_objectives,
        "objective_nodes_without_authored_mapping": [
            item["node_id"] for item in report["objective_mapping_review"]
        ],
    }

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
    report["requires_review"] += len(report["mapping_review"])
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

    knowledge_base, report = validate_nodes(
        knowledge_base,
        content_skills,
        source_blocks=extract_blocks(PDF_PATH),
    )


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

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    progress_structure = build_content_progress_structure(content_skills)
    with open(
        PROGRESS_STRUCTURE_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            progress_structure,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Highlighted keywords:", len(highlighted_keywords))
    print("Saved:", HIGHLIGHTED_KEYWORDS_PATH)
    print("Saved:", PROGRESS_STRUCTURE_PATH)
