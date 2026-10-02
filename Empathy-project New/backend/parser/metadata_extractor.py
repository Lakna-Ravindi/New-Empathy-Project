from pathlib import Path

import fitz


def extract_blocks(pdf_path):
    """Extract logical paragraphs and activities from PDF text blocks.

    PyMuPDF exposes spans and visual lines, but a visual line is not a
    knowledge-base unit. PDF blocks are the closest paragraph boundary. The
    small bullet parser below also preserves separate list items when a block
    contains an objectives or activity list.
    """
    pdf_path = Path(pdf_path)

    blocks = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document):
            page_data = page.get_text("dict")

            for block_index, block in enumerate(page_data["blocks"]):
                if "lines" not in block:
                    continue

                lines = []
                for line_index, line in enumerate(block["lines"]):
                    spans = [span for span in line["spans"] if span["text"].strip()]
                    if not spans:
                        continue

                    lines.append({
                        "line_index": line_index,
                        "text": _join_spans(spans),
                        "font_size": spans[0]["size"],
                        "font_name": spans[0]["font"],
                    })

                for item_index, item in enumerate(_logical_items(lines)):
                    blocks.append({
                        "page": page_number + 1,
                        "block_index": block_index,
                        "item_index": item_index,
                        "line_index": item["line_index"],
                        "text": item["text"],
                        "font_size": item["font_size"],
                        "font_name": item["font_name"],
                    })

    return _merge_continuations(blocks)


def _join_spans(spans):
    text = ""
    for span in spans:
        value = span["text"].strip()
        if not value:
            continue
        if text and not value[0] in ",.;:!?)]}":
            text += " "
        text += value
    return text.strip()


def _logical_items(lines):
    """Join wrapped lines, while keeping bullet items as separate blocks."""
    items = []
    current = None
    bullet_pending = False

    def flush():
        nonlocal current
        if current and current["text"].strip():
            items.append(current)
        current = None

    for line in lines:
        text = line["text"].strip()
        if _is_bullet(text):
            flush()
            bullet_pending = True
            continue

        bullet_text = _strip_bullet(text)
        starts_bullet = bullet_text != text
        if starts_bullet:
            flush()
            text = bullet_text
            bullet_pending = True

        if current is None:
            current = {
                "line_index": line["line_index"],
                "text": text,
                "font_size": line["font_size"],
                "font_name": line["font_name"],
            }
        else:
            current["text"] = _join_text(current["text"], text)
        bullet_pending = False

    flush()
    return items


def _is_bullet(text):
    return text in {"-", "•", "●", "\uf0b7"}


def _strip_bullet(text):
    return text.lstrip("-•●\uf0b7 ").strip()


def _join_text(left, right):
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return left[:-1] + right
    if right[0] in ",.;:!?)]}":
        return left + right
    return f"{left} {right}"


def _merge_continuations(blocks):
    """Join paragraphs split by a page break when the next block continues a sentence."""
    merged = []
    for block in blocks:
        if merged and _is_continuation(merged[-1]["text"], block["text"]):
            merged[-1]["text"] = _join_text(merged[-1]["text"], block["text"])
            continue
        merged.append(block)
    return merged


def _is_continuation(previous, current):
    previous = previous.strip()
    current = current.strip()
    if not previous or not current or not current[0].islower():
        return False
    if previous[-1] in ".!?;:)\"'":
        return False
    return previous[-1].isalnum() or previous[-1] in ",-"


if __name__ == "__main__":
    pdf = Path(__file__).resolve().parents[2] / "data" / "SEEK_Learning.pdf"

    result = extract_blocks(pdf)

    for item in result[:10]:
        print(item)