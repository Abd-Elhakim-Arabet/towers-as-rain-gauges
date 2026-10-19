"""Inline the Algeria map's data.json into its index.template.html → index.html (self-contained page).

    python scripts/build_site.py
"""

from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "site" / "algeria"
html = (PAGE / "index.template.html").read_text()
data = (PAGE / "data.json").read_text().replace("</", "<\\/")
(PAGE / "index.html").write_text(html.replace("/*DATA*/null", data))
print(f"wrote {PAGE / 'index.html'} ({(PAGE / 'index.html').stat().st_size / 1e3:.0f} kB)")
