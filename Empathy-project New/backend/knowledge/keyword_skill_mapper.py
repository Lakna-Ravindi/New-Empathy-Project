import re


def normalize_text(text):
    """Normalise text so PDF spacing and line breaks do not affect matching."""
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_node_for_keyword(keyword, nodes):
    """
    Find the most suitable knowledge node containing the highlighted keyword.
    Only nodes on the same PDF page are considered.
    """
    keyword_text = normalize_text(keyword["text"])

    matching_nodes = [
        node
        for node in nodes
        if node.get("page") == keyword.get("page")
        and keyword_text in normalize_text(node.get("content", ""))
    ]

    if not matching_nodes:
        return None

    # Prefer the shortest matching node because it is usually more specific.
    return min(
        matching_nodes,
        key=lambda node: len(normalize_text(node.get("content", "")))
    )


def find_ancestor_by_type(node, nodes_by_id, target_type):
    """Move up the hierarchy until a node of the requested type is found."""
    current = node

    while current:
        if current.get("type") == target_type:
            return current

        parent_id = current.get("parent_id")
        current = nodes_by_id.get(parent_id)

    return None


def map_keywords_to_skills(highlighted_keywords, knowledge_base):
    """
    Map each highlighted PDF keyword to:
    - its source knowledge node
    - its parent topic
    - its parent chapter / empathy skill
    """
    nodes_by_id = {
        node["id"]: node
        for node in knowledge_base
    }

    mapped_keywords = []

    for keyword in highlighted_keywords:
        source_node = find_node_for_keyword(
            keyword,
            knowledge_base
        )

        if source_node is None:
            mapped_keywords.append({
                "keyword": keyword["text"],
                "page": keyword["page"],
                "mapping_status": "not_found",
                "source_node_id": None,
                "topic": None,
                "skill_id": None,
                "skill": None
            })
            continue

        topic = find_ancestor_by_type(
            source_node,
            nodes_by_id,
            "topic"
        )

        skill = find_ancestor_by_type(
            source_node,
            nodes_by_id,
            "chapter"
        )

        mapped_keywords.append({
            "keyword": keyword["text"],
            "page": keyword["page"],
            "mapping_status": "mapped",

            "source_node_id": source_node["id"],
            "source_node_type": source_node["type"],
            "source_node_title": source_node["title"],

            "topic_id": topic["id"] if topic else None,
            "topic": topic["title"] if topic else None,

            "skill_id": skill["id"] if skill else None,
            "skill": skill["title"] if skill else None,

            "source": keyword.get("source")
        })

    return mapped_keywords