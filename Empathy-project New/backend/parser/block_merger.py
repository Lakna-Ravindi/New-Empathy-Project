def merge_spans(spans):
    merged = []
    current = None

    def join_text(left, right):
        left = left.rstrip()
        right = right.lstrip()

        if not left:
            return right

        if not right:
            return left

        if left.endswith("-"):
            return left[:-1] + right

        if right[0] in ",.;:!?)]}":
            separator = ""
        else:
            separator = " "

        return f"{left}{separator}{right}"

    for span in spans:
        text = span["text"].strip()

        if text == "":
            continue

        if current is None:
            current = span.copy()
            continue

        same_page = current["page"] == span["page"]
        same_block = current.get("block_index") == span.get("block_index")
        same_font = current["font_name"] == span["font_name"]
        same_size = current["font_size"] == span["font_size"]

        if same_page and same_block and same_font and same_size:
            current["text"] = join_text(current["text"], text)
        else:
            merged.append(current)
            current = span.copy()

    if current:
        merged.append(current)

    return merged