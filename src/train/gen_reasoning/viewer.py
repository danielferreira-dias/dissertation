"""Generate an HTML viewer for image descriptions."""

import json
import sys
from pathlib import Path


def generate_viewer(jsonl_path: Path, output_path: Path):
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            entries.append(json.loads(line))

    matched = sum(1 for e in entries if e.get("label_match", True))
    mismatched = len(entries) - matched

    rows = ""
    for e in entries:
        abs_path = Path(e["image_path"]).resolve()
        label_match = e.get("label_match", True)
        mismatch_class = "mismatch" if not label_match else ""
        match_badge = '<span class="match-badge bad">MISMATCH</span>' if not label_match else '<span class="match-badge good">MATCH</span>'
        rows += f"""
        <div class="card {mismatch_class}">
            <img src="file://{abs_path}" alt="{e['label']}">
            <div class="info">
                <div class="badges">
                    <span class="label">{e["label"]}</span>
                    {match_badge}
                    <span class="confidence {e.get("confidence", "unknown")}">Confidence: {e.get("confidence", "N/A")}</span>
                    <span class="fst">Fitzpatrick: {e.get("fitzpatrick_skin_type", "N/A")}</span>
                </div>
                <p>{e.get("reasoning", e.get("clinical_reasoning", "No reasoning"))}</p>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dermatology Descriptions Viewer</title>
<style>
    body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #e0e0e0; padding: 20px; }}
    h1 {{ text-align: center; color: #fff; }}
    .stats {{ text-align: center; color: #888; margin-bottom: 30px; }}
    .badges {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    .match-badge {{ font-size: 12px; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
    .match-badge.good {{ background: #1b5e20; color: #a5d6a7; }}
    .match-badge.bad {{ background: #b71c1c; color: #ef9a9a; }}
    .card.mismatch {{ border: 2px solid #c62828; }}
    .card {{ display: flex; gap: 20px; background: #2a2a2a; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    .card img {{ width: 300px; height: 300px; object-fit: contain; border-radius: 8px; background: #111; }}
    .info {{ flex: 1; display: flex; flex-direction: column; gap: 8px; }}
    .label {{ font-size: 18px; font-weight: bold; color: #4fc3f7; text-transform: uppercase; }}
    .confidence {{ font-size: 14px; padding: 2px 8px; border-radius: 4px; width: fit-content; }}
    .confidence.high {{ background: #2e7d32; color: #fff; }}
    .confidence.medium {{ background: #f57f17; color: #fff; }}
    .confidence.low {{ background: #c62828; color: #fff; }}
    .fst {{ font-size: 14px; color: #aaa; }}
    p {{ line-height: 1.6; color: #ccc; }}
</style>
</head>
<body>
    <h1>Dermatology Descriptions</h1>
    <div class="stats">{len(entries)} images | {matched} matched | {mismatched} mismatched</div>
    {rows}
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Viewer saved to {output_path}")
    print(f"Open with: open {output_path}")


if __name__ == "__main__":
    jsonl = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/reasoning/descriptions.jsonl")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/reasoning/viewer.html")
    generate_viewer(jsonl, out)
