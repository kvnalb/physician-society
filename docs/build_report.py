"""Build a static HTML report from target_report.md + demo artifacts (Jinja2)."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "docs" / "templates"
OUT_PATH = PROJECT_ROOT / "docs" / "build" / "demo_report.html"


def main() -> None:
    narrative = (PROJECT_ROOT / "docs" / "target_report.md").read_text(encoding="utf-8")
    summary = json.loads((PROJECT_ROOT / "artifacts" / "demo" / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((PROJECT_ROOT / "artifacts" / "demo" / "metrics.json").read_text(encoding="utf-8"))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("demo_report.html.j2")
    html = template.render(narrative=narrative, summary=summary, metrics=metrics)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
