from __future__ import annotations

from pathlib import Path

import fitz


def read_pdf(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)

    pages: list[dict] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document):
            pages.append({
                "page": page_number + 1,
                "text": page.get_text(),
            })

    return pages


if __name__ == "__main__":
    pdf = Path(__file__).resolve().parents[2] / "data" / "SEEK_Learning.pdf"
    result = read_pdf(pdf)

    if result:
        print(result[0])
    else:
        print({"page": 0, "text": ""})