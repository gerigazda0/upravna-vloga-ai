"""Iz PDF, DOCX, TXT in MD datotek potegne navadno besedilo."""
import io
from pathlib import Path


def extract_text(content: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".pdf":
            return _from_pdf(content)
        if suffix == ".docx":
            return _from_docx(content)
        if suffix in (".txt", ".md"):
            return content.decode("utf-8", errors="replace")
        # za vse ostalo pač probamo kot UTF-8
        return content.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"[Napaka pri branju datoteke {filename}: {exc}]"


def _from_pdf(content: bytes) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _from_docx(content: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(content))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
