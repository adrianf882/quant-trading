"""Extrae el texto plano de un archivo .epub (es un ZIP con XHTML adentro).

Uso:
    py tools/extract_epub.py "ruta/al/libro.epub" "reference/salida.txt"

No requiere dependencias externas: usa solo la libreria estandar.
"""
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return "\n".join(parser.parts)


def extract_epub(epub_path: Path, out_path: Path) -> int:
    chapters = []
    with zipfile.ZipFile(epub_path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm"))
        )
        for name in names:
            raw = zf.read(name).decode("utf-8", errors="ignore")
            chapters.append(f"\n\n===== {name} =====\n\n")
            chapters.append(_html_to_text(raw))
    text = "".join(chapters)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return len(text)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    epub, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = extract_epub(epub, out)
    print(f"Listo: {out} ({n} caracteres)")
