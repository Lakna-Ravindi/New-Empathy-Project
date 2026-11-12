from pathlib import Path
import re


def _load_fitz():
    try:
        import fitz
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PDF highlight extraction requires PyMuPDF, but its native "
            "_mupdf DLL could not be loaded. Repair the Windows Application "
            "Control policy/PyMuPDF installation before rebuilding."
        ) from exc
    return fitz

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "this",
    "to", "we", "what", "when", "with", "you", "your", "will", "able",
    "apply", "define", "describe", "discuss", "explain", "identify", "name",
    "recognize", "use", "understand", "learn", "develop", "practice",
}
GENERIC_CONCEPT_WORDS = {
    "being", "different", "importance", "nature", "others", "role",
    "three", "tools", "using", "well", "worldly",
}


def _highlight_rectangles(annotation):
    """Convert a highlight annotation's quadrilaterals into rectangles."""
    fitz = _load_fitz()
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
    fitz = _load_fitz()
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
                                "page": page_number,
                                "extraction_method": "pdf_annotation",
                            }
                        })

                annotation = annotation.next

    return keywords


def extract_keyword_candidates(
    nodes: list[dict],
    content_skills: list[dict] | None = None,
) -> list[dict]:
    """Extract source-backed concepts when explicit PDF highlights are sparse."""
    concepts: set[str] = set()
    for skill in content_skills or []:
        concepts.update(
            str(concept).strip()
            for concept in skill.get("key_concepts", [])
            if str(concept).strip()
        )
        for objective in skill.get("objectives", []):
            title = str(objective.get("objective_title", "")).strip()
            words = [
                word.lower()
                for word in re.findall(r"[a-z]+", title.lower())
                if (
                    word not in STOP_WORDS
                    and word not in GENERIC_CONCEPT_WORDS
                    and len(word) > 4
                )
            ]
            if 2 <= len(words) <= 4:
                concepts.add(" ".join(words))

    candidates = []
    seen = set()
    for node in nodes:
        if node.get("type") not in {"content", "learning_objective"}:
            continue
        node_text = re.sub(
            r"\s+",
            " ",
            f"{node.get('title', '')} {node.get('content', '')}"
        ).strip().lower()
        for concept in sorted(concepts, key=lambda value: (-len(value), value.lower())):
            normalized_concept = re.sub(r"\s+", " ", concept).strip().lower()
            if (
                not normalized_concept
                or normalized_concept in seen
                or normalized_concept not in node_text
            ):
                continue
            seen.add(normalized_concept)
            candidates.append({
                "text": concept,
                "page": node.get("page"),
                "type": "keyword_candidate",
                "source": {
                    "document": "SEEK Learning",
                    "page": node.get("page"),
                    "node_id": node.get("id"),
                    "extraction_method": "source_concept",
                },
            })

    return candidates