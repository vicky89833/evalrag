"""Upload the canonical PDF (or any doc), then print the doc_id.

Usage:
    python scripts/seed_demo_doc.py [path-to-doc]

Defaults to evals/canonical.pdf if no path provided.
"""
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = ROOT / "evals" / "canonical.pdf"


def main() -> int:
    doc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC
    if not doc_path.exists():
        print(f"missing: {doc_path}")
        print("Drop a PDF/DOCX/MD/TXT at evals/canonical.pdf or pass a path.")
        return 2
    with doc_path.open("rb") as f:
        r = httpx.post(
            "http://localhost:8000/docs",
            files={"file": (doc_path.name, f.read(), "application/octet-stream")},
            timeout=300,
        )
    r.raise_for_status()
    doc = r.json()
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
