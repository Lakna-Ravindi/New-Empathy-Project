from pathlib import Path

import fitz


def extract_blocks(pdf_path):
    pdf_path = Path(pdf_path)

    blocks = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document):
            page_data = page.get_text("dict")

            for block_index, block in enumerate(page_data["blocks"]):
                if "lines" in block:
                    for line_index, line in enumerate(block["lines"]):
                        for span_index, span in enumerate(line["spans"]):
                            text = span["text"].strip()
                            if text == "":
                                continue

                            blocks.append({
                                "page": page_number + 1,
                                "block_index": block_index,
                                "line_index": line_index,
                                "span_index": span_index,
                                "text": text,
                                "font_size": span["size"],
                                "font_name": span["font"],
                            })

    return blocks


def merge_spans(spans):
    merged = []
    current = None

    def _join_text(left, right):
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
            current["text"] = _join_text(current["text"], text)
        else:
            merged.append(current)
            current = span.copy()

    if current:
        merged.append(current)

    return merged


if __name__ == "__main__":
    pdf = Path(__file__).resolve().parents[2] / "data" / "SEEK_Learning.pdf"

    result = extract_blocks(pdf)
    merged = merge_spans(result)

    for item in merged[:10]:
        print(item)