import json
import hashlib
from pathlib import Path
import re
from time import perf_counter
from time import time

from app.agents.coverage import build_store, load_clauses
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "corpus" / "source"
PROCESSED_PATH = ROOT / "data" / "corpus" / "processed" / "indexed_corpus.json"


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def category_for(text: str) -> str:
    lowered = text.lower()
    if any(
        term in lowered
        for term in (
            "exclusion",
            "excluded",
            "not covered",
            "will not pay",
            "shall not be liable",
            "not indemnify",
        )
    ):
        return "exclusion"
    if any(
        term in lowered
        for term in (
            "limitation as to use",
            "permitted use",
            "restriction",
            "hire or reward",
            "racing",
            "commercial use",
            "delivery service",
            "carrying goods",
        )
    ):
        return "restriction"
    if any(term in lowered for term in ("claim procedure", "notify the insurer", "claim intimation")):
        return "procedure"
    return "coverage"


def parse_source_pdfs() -> list[dict[str, str]]:
    clauses = []
    for path in sorted(SOURCE_DIR.glob("*.pdf")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk_number, chunk in enumerate(split_text(text), start=1):
                if not chunk:
                    continue
                clauses.append(
                    {
                        "clause_id": f"PDF-{digest}-P{page_number:04d}-C{chunk_number:03d}",
                        "source": path.name,
                        "category": category_for(chunk),
                        "topics": path.stem,
                        "text": chunk,
                    }
                )
    return clauses


def main() -> None:
    started = perf_counter()
    pdf_clauses = parse_source_pdfs()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.write_text(
        json.dumps({"clauses": pdf_clauses}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    clauses = load_clauses()
    build_store()
    freshness = {
        "built_at_epoch": time(),
        "source": "data/corpus/source/*.pdf",
        "clause_count": len(clauses),
        "pdf_chunk_count": len(pdf_clauses),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    Path("data/index/freshness.json").write_text(
        json.dumps(freshness, indent=2), encoding="utf-8"
    )
    print(json.dumps(freshness, indent=2))


if __name__ == "__main__":
    main()
