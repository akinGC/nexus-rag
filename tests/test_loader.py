import tempfile
import unittest
from pathlib import Path

from backend.ingestion.loader import load_document


class LoaderTests(unittest.TestCase):
    def test_markdown_preserves_hierarchy_and_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "handbook.md"
            path.write_text("# Employee Handbook\n## Leave\n### California\nEligible employees receive 12 weeks.")
            document = load_document(path, ["hr_admin"], "HR")
        self.assertEqual(document.title, "Employee Handbook")
        self.assertIn("### California", document.pages[0][1])
        self.assertEqual(document.allowed_roles, ["hr_admin"])
        self.assertEqual(document.department, "HR")

    def test_empty_document_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("")
            with self.assertRaises(ValueError):
                load_document(path, ["employee"])