from quality_intelligence import pdf_loader
from quality_intelligence.pdf_loader import extract_text_with_pdftotext, load_pdf


class FakePage:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self) -> str:
        return self.text


def test_load_pdf_skips_broken_pdf_metadata(monkeypatch, tmp_path):
    pdf_path = tmp_path / "broken-metadata.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    class BrokenMetadataReader:
        pages = [FakePage("Document body")]

        def __init__(self, path: str):
            self.path = path

        @property
        def metadata(self):
            raise RuntimeError("metadata object is encrypted")

    monkeypatch.setattr(pdf_loader, "PdfReader", BrokenMetadataReader)

    document = load_pdf(pdf_path)

    assert document.title is None
    assert document.author is None
    assert document.pages[0].text == "Document body"


def test_load_pdf_uses_pdftotext_when_pypdf_fails(monkeypatch, tmp_path):
    pdf_path = tmp_path / "encrypted.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    class FailingReader:
        def __init__(self, path: str):
            raise RuntimeError("cryptography is required for AES algorithm")

    monkeypatch.setattr(pdf_loader, "PdfReader", FailingReader)
    monkeypatch.setattr(pdf_loader, "extract_text_with_pdftotext", lambda path: "First page\fSecond page")
    monkeypatch.setattr(pdf_loader, "extract_text_with_ocrmypdf", lambda path: "")

    document = load_pdf(pdf_path, ocr_fallback=True)

    assert [page.text for page in document.pages] == ["First page", "Second page"]


def test_extract_text_with_pdftotext_returns_stdout(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    class Completed:
        returncode = 0
        stdout = "Extracted text"
        stderr = ""

    def fake_run(command, **kwargs):
        assert command[-1] == "-"
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return Completed()

    monkeypatch.setattr(pdf_loader, "_pdftotext_command", lambda: "pdftotext")
    monkeypatch.setattr(pdf_loader.subprocess, "run", fake_run)

    assert extract_text_with_pdftotext(pdf_path) == "Extracted text"
