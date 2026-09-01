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


if __name__ == "__main__":
    pdf = Path(__file__).resolve().parents[2] / "data" / "SEEK_Learning.pdf"

    result = extract_blocks(pdf)

    for item in result[:10]:
        print(item)