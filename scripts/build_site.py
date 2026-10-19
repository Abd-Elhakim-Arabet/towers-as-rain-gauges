"""Inline site/data.json into site/index.template.html → site/index.html (one self-contained page)."""

from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
html = (SITE / "index.template.html").read_text()
data = (SITE / "data.json").read_text().replace("</", "<\\/")
(SITE / "index.html").write_text(html.replace("/*DATA*/null", data))
print(f"wrote {SITE / 'index.html'} ({(SITE / 'index.html').stat().st_size / 1e3:.0f} kB)")
