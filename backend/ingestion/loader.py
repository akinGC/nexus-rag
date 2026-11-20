from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import re


@dataclass
class Document:
	title: str
	source: str
	source_type: str
	pages: list[tuple[int | None, str]]
	allowed_roles: list[str] = field(default_factory=lambda: ["employee"])
	department: str = "General"
	document_id: str = field(default_factory=lambda: uuid4().hex)


def load_document(path: Path, roles: list[str], department: str = "General") -> Document:
	suffix = path.suffix.lower()
	if suffix == ".pdf":
		from pypdf import PdfReader
		pages = [(number, page.extract_text() or "") for number, page in enumerate(PdfReader(path).pages, 1)]
	elif suffix == ".docx":
		from docx import Document as WordDocument
		word = WordDocument(path)
		text = "\n".join(
			f"{'#' * min(int(paragraph.style.name.split()[-1]), 6)} {paragraph.text}"
			if paragraph.style.name.startswith("Heading ") and paragraph.style.name.split()[-1].isdigit()
			else paragraph.text for paragraph in word.paragraphs
		)
		tables = ["\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows) for table in word.tables]
		pages = [(None, "\n\n".join([text, *tables]))]
	elif suffix in {".txt", ".md", ".html", ".htm"}:
		text = path.read_text(encoding="utf-8")
		if suffix in {".html", ".htm"}:
			text = parse_html(text)[1]
		pages = [(None, text)]
	else:
		raise ValueError("Supported formats: PDF, DOCX, TXT, Markdown, HTML")
	title = path.stem.replace("_", " ")
	for _, text in pages:
		heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
		if heading:
			title = heading.group(1)
			break
	if not any(text.strip() for _, text in pages):
		raise ValueError("Document has no extractable text; scanned PDFs require OCR")
	return Document(title, path.name, suffix.lstrip("."), pages, roles, department)


def parse_html(html: str) -> tuple[str, str]:
	from bs4 import BeautifulSoup
	soup = BeautifulSoup(html, "html.parser")
	title = soup.title.get_text(strip=True) if soup.title else "Web document"
	for element in soup(["script", "style", "nav", "footer"]):
		element.decompose()
	for heading in soup.find_all(re.compile(r"^h[1-6]$")):
		heading.replace_with(f"\n{'#' * int(heading.name[1])} {heading.get_text(' ', strip=True)}\n")
	return title, soup.get_text("\n", strip=True)