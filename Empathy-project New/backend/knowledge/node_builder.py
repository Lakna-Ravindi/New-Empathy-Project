def build_node(block, node_type, node_id, parent_id=None):

    node = {

        "id": node_id,

        "title": block["text"].strip()[:120],

        "type": node_type,

        "content": block["text"].strip(),

        "page": block["page"],

        "parent_id": parent_id

    }

    return node