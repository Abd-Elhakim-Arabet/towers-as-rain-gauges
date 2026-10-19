"""Inline each page's data.json into its index.template.html → index.html (self-contained pages).

    python scripts/build_site.py          # site/ and site/algeria/
"""

from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
for page in (SITE, SITE / "algeria"):
    html = (page / "index.template.html").read_text()
    data = (page / "data.json").read_text().replace("</", "<\\/")
    (page / "index.html").write_text(html.replace("/*DATA*/null", data))
    print(f"wrote {page / 'index.html'} ({(page / 'index.html').stat().st_size / 1e3:.0f} kB)")
