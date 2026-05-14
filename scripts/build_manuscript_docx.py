"""Split Manuscript_Draft.md and rebuild the three docx deliverables via pandoc.

Outputs:
  docs/HPV_manuscript.docx       — Abstract + Methods + Results + Discussion + References
  docs/HPV_supplementary.docx    — Supplementary materials block (figure + table list)
  docs/HPV_tables_figures.docx   — Main tables and figures legends block
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "Manuscript_Draft.md"

OUTPUTS = {
    "manuscript": ROOT / "docs" / "HPV_manuscript.docx",
    "tables_figures": ROOT / "docs" / "HPV_tables_figures.docx",
    "supplementary": ROOT / "docs" / "HPV_supplementary.docx",
}


def split_sections(text: str) -> dict[str, str]:
    """Return three sub-documents keyed by output name.

    The manuscript covers Abstract → Suggested Discussion outline + References.
    The tables_figures doc covers the "Main tables and figures" subsection.
    The supplementary doc covers the "Supplementary materials" subsection.
    """
    lines = text.splitlines(keepends=True)

    def find_h2(needle: str) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith("## ") and needle in ln:
                return i
        raise SystemExit(f"could not find heading: {needle}")

    def find_h3(needle: str) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith("### ") and needle in ln:
                return i
        raise SystemExit(f"could not find subheading: {needle}")

    title_end = next(i for i, ln in enumerate(lines) if ln.startswith("## "))
    tab_fig_h2 = find_h2("Tables and Figures")
    main_h3 = find_h3("Main tables and figures")
    supp_h3 = find_h3("Supplementary materials")
    discussion_h2 = find_h2("Suggested Discussion outline")
    references_h2 = find_h2("References")

    title_block = "".join(lines[:title_end])  # the H1 title line

    # Manuscript = title + everything Abstract..Statistical analysis +
    # Results + Suggested Discussion outline + References (skip the
    # tables/figures listing entirely; those land in their own deliverable).
    manuscript = (
        title_block
        + "".join(lines[title_end:tab_fig_h2])
        + "".join(lines[discussion_h2:])
    )

    tables_figures = (
        f"# Main tables and figures\n\n"
        + "".join(lines[main_h3 + 1:supp_h3]).lstrip("\n")
    )

    supplementary = (
        f"# Supplementary materials\n\n"
        + "".join(lines[supp_h3 + 1:discussion_h2]).lstrip("\n")
    )

    return {
        "manuscript": manuscript,
        "tables_figures": tables_figures,
        "supplementary": supplementary,
    }


def normalise(md: str) -> str:
    """Drop file-pointer notes that are noise in the docx output."""
    md = re.sub(r"\s*File:\s*`[^`]+`\.", ".", md)
    md = re.sub(r"\s*Files:\s*`[^`]+`(,\s*`[^`]+`)*\.", ".", md)
    md = re.sub(r"\s*Sources?:\s*`[^`]+`(,\s*`[^`]+`)*[^.]*\.", ".", md)
    return md


def run_pandoc(md: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "pandoc",
            "--from", "markdown",
            "--to", "docx",
            "--standalone",
            "--output", str(out_path),
        ],
        input=md,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"pandoc failed for {out_path.name}")
    print(f"wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    parts = split_sections(text)
    for key, path in OUTPUTS.items():
        run_pandoc(normalise(parts[key]), path)


if __name__ == "__main__":
    main()
