def build_node(block, node_type, node_id, parent_id=None):

    text = block.get("text", "").strip()

    node = {
        "id": node_id,
        "title": text[:120],
        "type": node_type,
        "content": text,
        "page": block.get("page"),
        "parent_id": parent_id
    }

    return node