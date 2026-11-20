import re

from backend.ingestion.loader import Document


def chunk_document(document: Document, size: int = 1200) -> list[dict]:
    if size < 100:
        raise ValueError("Chunk size must be at least 100 characters")
    chunks = []
    headings = []
    for page, text in document.pages:
        buffer = []

        def flush():
            content = "\n".join(buffer).strip()
            if not content:
                return
            for offset in range(0, len(content), size):
                chunks.append({
                    "chunk_id": f"{document.document_id}:{len(chunks)}",
                    "document_id": document.document_id, "title": document.title,
                    "source": document.source, "source_type": document.source_type,
                    "department": document.department, "allowed_roles": document.allowed_roles,
                    "page": page, "section": " / ".join(headings) or document.title,
                    "paragraph": len(chunks) + 1, "text": content[offset:offset + size],
                })
            buffer.clear()

        for line in text.splitlines():
            heading = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading:
                flush()
                level = len(heading.group(1))
                headings = headings[:level - 1]
                headings.append(heading.group(2))
            else:
                buffer.append(line)
        flush()
    return chunks
