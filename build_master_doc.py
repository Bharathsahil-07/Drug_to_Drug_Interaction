from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
OUTPUT = ROOT / "PROJECT_MASTER_DOC.md"

md_files = sorted(
    [p for p in ROOT.glob("*.md") if p.name != OUTPUT.name],
    key=lambda p: p.name.lower(),
)

py_files = sorted(
    [p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts],
    key=lambda p: str(p.relative_to(ROOT)).lower(),
)

lines = []
lines.append("# Project Master Document")
lines.append("")
lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("")
lines.append("## Overview")
lines.append("This file consolidates all top-level project markdown documentation, includes a full Python file index, and summarizes the model currently used by the API.")
lines.append("")

lines.append("## Model Used In This Project")
lines.append("")
lines.append("The serving model is a **Graph Attention Network (GAT) multi-task architecture**, implemented in `mt_gat_model.py` via class `DrugInteractionMTGAT`.")
lines.append("")
lines.append("### Architecture")
lines.append("- Encoder: 3 stacked `GATConv` layers")
lines.append("- Typical checkpoint config loaded by API: `hidden_dim=128`, `embedding_dim=64`, `heads=4`")
lines.append("- Pair representation: concatenation of two drug node embeddings")
lines.append("- Multi-task heads:")
lines.append("  - Interaction head: binary interaction logit")
lines.append("  - Severity head: 3-class logits (Minor, Moderate, Major)")
lines.append("  - Confidence head: scalar regression in `[0,1]`")
lines.append("")
lines.append("### Inference & Calibration")
lines.append("- API loads trained weights from `data/trained_model_v2.pt`")
lines.append("- Probability is temperature-calibrated in the API (`model_temperature = 1.15`) before displaying risk")
lines.append("- The UI label `Calibrated` indicates this post-processing step")
lines.append("")

lines.append("## Python Files Used In The Project")
lines.append("")
for p in py_files:
    rel = p.relative_to(ROOT).as_posix()
    lines.append(f"- `{rel}`")
lines.append("")

lines.append("## Combined Markdown Documentation")
lines.append("")
for md in md_files:
    rel = md.relative_to(ROOT).as_posix()
    lines.append(f"### Source: `{rel}`")
    lines.append("")
    try:
        content = md.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        content = f"[Error reading file: {exc}]"

    if content:
        lines.append(content)
    else:
        lines.append("[Empty file]")

    lines.append("")
    lines.append("---")
    lines.append("")

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUTPUT}")
print(f"Included markdown files: {len(md_files)}")
print(f"Included python files: {len(py_files)}")

