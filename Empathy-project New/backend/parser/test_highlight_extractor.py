import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from parser.highlight_extractor import (
    extract_highlighted_keywords,
    extract_keyword_candidates,
)


class KeywordCandidateTests(unittest.TestCase):
    def test_highlighted_keywords_are_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "highlighted.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Grounding")
            annotation = page.add_highlight_annot(fitz.Rect(70, 58, 145, 75))
            annotation.update()
            document.save(path)
            document.close()

            keywords = extract_highlighted_keywords(path)

        self.assertEqual(len(keywords), 1)
        self.assertEqual(keywords[0]["type"], "highlighted_keyword")
        self.assertEqual(keywords[0]["text"], "Grounding")

    def test_pdf_without_highlights_returns_no_manual_keywords(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plain.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Grounding")
            document.save(path)
            document.close()

            self.assertEqual(extract_highlighted_keywords(path), [])

    def test_candidates_are_source_backed_and_deduplicated(self):
        nodes = [{
            "id": "node-1",
            "type": "content",
            "title": "Grounding and body literacy",
            "content": "Grounding supports body literacy.",
            "page": 1,
        }]
        skills = [{
            "key_concepts": ["Grounding", "Body Literacy", "the"],
            "objectives": [{
                "objective_title": "Identify grounding practices",
            }],
        }]

        candidates = extract_keyword_candidates(nodes, skills)

        self.assertEqual(
            {candidate["text"].lower() for candidate in candidates},
            {"grounding", "body literacy"},
        )
        self.assertTrue(all(
            candidate["type"] == "keyword_candidate"
            for candidate in candidates
        ))

    def test_absent_concepts_are_not_invented(self):
        candidates = extract_keyword_candidates(
            [{
                "id": "node-1",
                "type": "content",
                "title": "A neutral paragraph",
                "content": "This text has no authored concept.",
                "page": 1,
            }],
            [{"key_concepts": ["Grounding"], "objectives": []}],
        )

        self.assertEqual(candidates, [])
