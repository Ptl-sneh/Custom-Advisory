import os
import fitz
import docx
import openpyxl
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from schemas import DocType
from logger import get_logger

logger = get_logger(__name__)


class ParsedDocument(BaseModel):
    filename: str
    file_path: str
    raw_text: str  # full extracted text, passed to chunker
    page_count: Optional[int] = None  # PDF only; None for DOCX/XLSX
    file_size_kb: float
    doc_type: DocType  # enum from schemas.py
    source_name: str
    issuing_authority: Optional[str] = None
    issue_date: Optional[str] = None
    reference_number: Optional[str] = None
    tags: list[str] = []


def parse_pdf(file_path: str) -> tuple[str, None]:

    logger.info(f"Parsing PDF | file={Path(file_path).name}")
    try:
        doc = fitz.open(file_path)
        pages_text = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                # Add page number in prefix for citations
                pages_text.append(f"[PAGE {page_num}]\n{text.strip()}")

        page_count = len(doc)
        doc.close()

        logger.debug(
            f"PDF parsed | pages={page_count} | chars={sum(len(p) for p in pages_text)}"
        )
        return "\n\n".join(pages_text), page_count

    except Exception as e:
        logger.info(f"PDF parse failed | file={Path(file_path).name} | error={str(e)}")
        raise


def parse_docx(file_path: str) -> tuple[str, None]:

    logger.info(f"Parsing DOCX | file={Path(file_path).name}")
    try:
        document = docx.Document(file_path)
        paragraphs = []

        #  Extract paragraphs
        for para in document.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())

        # Extract tables (if any)
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    paragraphs.append(row_text)

        logger.debug(f"DOCX parsed | paragraphs={len(paragraphs)}")
        return "\n\n".join(paragraphs), None

    except Exception as e:
        logger.info(f"DOCX parse failed | file={Path(file_path).name} | error={str(e)}")
        raise


def parse_xlsx(file_path: str) -> tuple[str, None]:

    logger.info(f"Parsing XLSX | file={Path(file_path).name}")
    try:

        # read_only=True → streams file (memory efficient for large tariff schedules)
        # data_only=True → reads computed values, not raw formulas

        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        all_text = []
        total_rows = 0

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            all_text.append(f"[SHEET: {sheet_name}]")  # sheet tag for citation context

            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join(
                    str(cell).strip()
                    for cell in row
                    if cell is not None and str(cell).strip()
                )
                if row_text:
                    all_text.append(row_text)
                    total_rows += 1

        sheet_count = len(wb.sheetnames)
        wb.close()
        logger.debug(f"XLSX parsed | sheets={sheet_count} | rows={total_rows}")
        return "\n\n".join(all_text), None

    except Exception as e:
        logger.info(f"XLSX parse failed | file={Path(file_path).name} | error={str(e)}")
        raise


def parse_document(file_path: str, metadata: dict) -> ParsedDocument:

    path = Path(file_path)
    ext = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_kb = os.path.getsize(file_path) / 1024
    logger.info(f"Document parse started | file={path.name} | size={size_kb:.1f}KB")

    try:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if ext == ".pdf":
            raw_text, page_count = parse_pdf(file_path)
        elif ext == ".docx":
            raw_text, page_count = parse_docx(file_path)
        elif ext in (".xlsx", ".xls"):
            raw_text, page_count = parse_xlsx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if not raw_text.strip():
            raise ValueError(f"No text extracted from: {path.name}")

        logger.info(
            f"Document parse complete | file={path.name} | chars={len(raw_text)}"
        )

        return ParsedDocument(
            filename=path.name,
            file_path=str(file_path),
            raw_text=raw_text,
            page_count=page_count,
            file_size_kb=round(size_kb, 2),
            doc_type=metadata.get("doc_type", DocType.OTHER),
            source_name=metadata.get("source_name", path.stem),
            issuing_authority=metadata.get("issuing_authority"),
            issue_date=metadata.get("issue_date"),
            reference_number=metadata.get("reference_number"),
            tags=metadata.get("tags", []),
        )

    except Exception as e:
        logger.info(f"Document parse failed | file={path.name} | error={str(e)}")
        raise
