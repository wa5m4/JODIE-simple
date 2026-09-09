"""Extract text from the two PDFs in 论文/ for review."""
import sys

def extract(path, out_path):
    text = None
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(pages)
    except ImportError:
        pass
    if text is None:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n\n".join(pages)
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(path)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n\n".join(pages)
            except ImportError:
                print(f"no pdf library available for {path}")
                return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{path} -> {out_path} ({len(text)} chars)")

if __name__ == "__main__":
    base = r"c:\Users\17789\Desktop\jodie-simple-refactored\论文"
    extract(base + r"\dynahb.pdf", base + r"\dynahb.txt")
    extract(base + r"\科研过程-第三阶段.pdf", base + r"\科研过程-第三阶段.txt")
