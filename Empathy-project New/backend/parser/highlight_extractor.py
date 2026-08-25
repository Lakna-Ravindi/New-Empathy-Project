from pathlib import Path
import fitz


def _highlight_rectangles(annotation):
    """Convert a highlight annotation's quadrilaterals into rectangles."""
    vertices = annotation.vertices
    rectangles = []

    for index in range(0, len(vertices), 4):
        quad = fitz.Quad(vertices[index:index + 4])
        rectangles.append(quad.rect)

    return rectangles


def extract_highlighted_keywords(pdf_path: str | Path) -> list[dict]:
    """
    Extract highlighted words and phrases from a PDF.

    Returns page number, keyword text, and its bounding position.
    """
    pdf_path = Path(pdf_path)
    keywords = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            annotation = page.first_annot

            while annotation:
                annotation_type = annotation.type[0]

                if annotation_type == fitz.PDF_ANNOT_HIGHLIGHT:
                    rectangles = _highlight_rectangles(annotation)

                    # Gets text accurately from each separate highlight area.
                    text_parts = [
                        page.get_textbox(rect).strip()
                        for rect in rectangles
                    ]

                    text = " ".join(part for part in text_parts if part)

                    if text:
                        keywords.append({
                            "text": text,
                            "page": page_number,
                            "type": "highlighted_keyword",
                            "source": {
                                "document": "SEEK Learning",
                                "page": page_number
                            }
                        })

                annotation = annotation.next

    return keywords