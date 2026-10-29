from __future__ import annotations

import io
import logging
import re
from datetime import date, timedelta
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from .domain import Project, require_tgo_url

BASE = "https://tver.tgo.or.th"
METHOD_PATTERN = r"T-VER-(?:[SP]-METH-\d{2}-\d{2}|METH-(?:FOR|AGR)-\d{2})"


def number(text: str | float | int | None) -> float | None:
    if text is None:
        return None
    value = str(text).strip().translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    if value in ("", "-", "N/A", "nan", "None"):
        return None
    value = value.replace(",", "").replace("\u00a0", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        raise ValueError(f"Invalid numeric field: {text!r}")
    return float(value)


def thai_date(text: str | None) -> str | None:
    if not text or not text.strip() or text.strip() == "-":
        return None
    match = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*", text)
    if not match:
        raise ValueError(f"Unsupported official date format: {text!r}")
    day, month, year = map(int, match.groups())
    if year >= 2400:
        year -= 543
    return date(year, month, day).isoformat()


def fiscal_period(year_be: int) -> dict:
    year_ce = year_be - 543 if year_be >= 2400 else year_be
    return {"fiscal_year_be": year_be, "fiscal_year_ce": year_ce,
            "period_start": f"{year_ce - 1}-10-01", "period_end": f"{year_ce}-09-30"}


def table_fields(table) -> dict[str, str]:
    fields = {}
    for row in table.select("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) == 2:
            key = cells[0].get_text(" ", strip=True)
            value = cells[1].get_text(" ", strip=True)
            fields[key] = value
    return fields


def parse_csv(raw: bytes, source: dict, kind: str) -> tuple[list[dict], dict]:
    # UTF-8 BOM and Thai Windows encodings both occur in public government files.
    text = None
    for encoding in ("utf-8-sig", "cp874"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("Unsupported CSV encoding")
    frame = pd.read_csv(io.StringIO(text), dtype=str).dropna(how="all")
    category_column = "T-VER Project Type" if kind == "trading" else "Project_Type"
    if category_column not in frame or "FY" not in frame:
        raise ValueError(f"CSV schema changed: {list(frame.columns)}")
    eligible = frame[frame[category_column].str.strip() == "FOR&AGR"]
    rows = []
    for index, row in eligible.iterrows():
        fy = int(number(row["FY"]))
        result = {"kind": kind, "category": "FOR&AGR", **fiscal_period(fy),
                  "source_id": source["source_id"], "source_url": source["url"],
                  "source_row": int(index) + 2, "original_values": row.to_dict()}
        if kind == "trading":
            prefixes = {"volume_tonnes": "Volume", "value_baht": "Value",
                        "minimum_price_baht_per_tonne": "Minimum", "maximum_price_baht_per_tonne": "Maximum",
                        "reported_weighted_price": "Weigh"}
            for field, prefix in prefixes.items():
                matches = [c for c in frame if c.strip().startswith(prefix)]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous CSV column for {field}")
                result[field] = number(row[matches[0]])
            if not result["volume_tonnes"] or any(result[k] is None or result[k] < 0 for k in prefixes):
                raise ValueError("Invalid trading quantities")
            result["calculated_weighted_price"] = result["value_baht"] / result["volume_tonnes"]
            if abs(result["calculated_weighted_price"] - result["reported_weighted_price"]) > 0.02:
                raise ValueError("Weighted price does not reconcile with value / volume")
            if not result["minimum_price_baht_per_tonne"] <= result["calculated_weighted_price"] <= result["maximum_price_baht_per_tonne"]:
                raise ValueError("Weighted price outside reported range")
            result["unit_warning"] = "Official volume header says baht; interpreted as tonnes because value/volume reconciles with per-tonne price."
        else:
            column = "GHG_Reduction_Credit" if kind == "standard_issuance" else "GHG_Reduction_Expected"
            if column not in row or row.get("GHG_Unit") != "tCO2eq":
                raise ValueError("Missing or incompatible aggregate amount/unit")
            result["tonnes"] = number(row[column])
            if result["tonnes"] is None or result["tonnes"] < 0:
                raise ValueError("Negative or missing aggregate")
        rows.append(result)
    if not rows or len({r["fiscal_year_be"] for r in rows}) != len(rows):
        raise ValueError("Missing FOR&AGR records or duplicate fiscal years")
    return rows, {"nonempty_rows": len(frame), "eligible_rows": len(rows),
                  "excluded_rows": len(frame) - len(rows), "kind": kind,
                  "scope": "Mixed Category 13 aggregate; not removal-only or project-specific"}


def listing_parameters(raw: bytes, program: int, group: int) -> dict:
    soup = BeautifulSoup(raw, "html.parser")
    options = soup.select("#filter_category_id option")
    matches = [option for option in options if "[FOR&AGR]" in option.get_text()]
    if len(matches) != 1:
        raise ValueError("Cannot resolve actual Category 13 form option")
    return {"filter_program_id": str(program), "filter_group_id": str(group),
            "filter_lang": "th", "filter_category_id": matches[0]["value"],
            "filter_size_id": "", "filter_status_id": ""}


def parse_listing(raw: bytes, source: dict) -> list[dict]:
    soup = BeautifulSoup(raw, "html.parser")
    result = []
    table = soup.select_one("#grid-data") or soup.find("table")
    if not table:
        raise ValueError("Missing project listing table")
    for index, row in enumerate(table.select("tbody tr")):
        cells = row.find_all("td", recursive=False)
        if len(cells) != 10:
            if "ไม่พบ" in row.get_text():
                continue
            raise ValueError(f"Project listing column count changed: {len(cells)}")
        values = [c.get_text(" ", strip=True) for c in cells]
        if "[FOR&AGR]" not in values[5]:
            raise ValueError("Server returned a project outside FOR&AGR")
        link = cells[2].find("a", href=True)
        match = re.search(r"/project/(\d+)/(\d+)/(\d+)", link["href"] if link else "")
        if not match:
            raise ValueError("Unrecognized project link")
        program, group, project_id = match.groups()
        cycle_amounts = {str(int(n)): number(v) for n, v in re.findall(r"(\d+)\s*:\s*([\d,]+(?:\.\d+)?)", values[7])}
        result.append({"project_id": project_id, "registration_number": values[1],
                       "program_id": int(program), "group_id": int(group),
                       "source_url": require_tgo_url(urljoin(BASE, link["href"])),
                       "name": values[2], "expected_annual_tonnes": number(values[6]),
                       "certified_total_tonnes": number(values[8]), "certified_cycles": cycle_amounts,
                       "source_id": source["source_id"], "source_row": index + 1,
                       "original_values": values})
    return result


def parse_methods(raw: bytes, source: dict, program: int) -> list[dict]:
    soup = BeautifulSoup(raw, "html.parser")
    result = []
    for table_index, table in enumerate(soup.select("table")):
        for row_index, row in enumerate(table.select("tbody tr")):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 3:
                continue
            code = cells[0].get_text(" ", strip=True)
            if not re.fullmatch(r"T-VER-[SP]-METH-13-\d{2}", code):
                continue
            result.append({"code": code, "version": cells[1].get_text(" ", strip=True),
                           "title": cells[2].get_text(" ", strip=True), "program_id": program,
                           "family_code": code, "is_historical": False,
                           "detail_url": require_tgo_url(urljoin(BASE, cells[2].find("a", href=True)["href"])) if cells[2].find("a", href=True) else None,
                           "links": [require_tgo_url(urljoin(BASE, a["href"])) for a in row.select("a[href]") if a["href"].lower().endswith(".pdf")],
                           "source_id": source["source_id"], "source_url": source["url"],
                           "locator": f"table[{table_index}]/row[{row_index}]"})
    if not result:
        raise ValueError("No Category 13 methodology definitions parsed")
    return result


def parse_method_history(raw: bytes, source: dict, family: dict) -> list[dict]:
    soup = BeautifulSoup(raw, "html.parser")
    history = []
    for table_index, table in enumerate(soup.select("table")):
        for row_index, row in enumerate(table.select("tbody tr")):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 3:
                continue
            code = cells[0].get_text(" ", strip=True)
            if not re.fullmatch(METHOD_PATTERN, code):
                continue
            # Only a version table reached from a fetched Category 13 definition establishes this alias.
            history.append({"code": code, "version": cells[1].get_text(" ", strip=True),
                            "validity_original": cells[2].get_text(" ", strip=True),
                            "title": family["title"], "family_code": family["code"],
                            "program_id": family["program_id"], "is_historical": True,
                            "detail_url": source["url"],
                            "links": [require_tgo_url(urljoin(BASE, a["href"])) for a in row.select("a[href]") if a["href"].lower().endswith(".pdf")],
                            "source_id": source["source_id"], "source_url": source["url"],
                            "locator": f"table[{table_index}]/row[{row_index}]"})
    if not history or family["code"] not in {m["code"] for m in history}:
        raise ValueError("Method history does not contain its Category 13 family code")
    return history


def parse_project(raw: bytes, source: dict, listing: dict) -> tuple[Project, list[dict], list[dict], list[dict]]:
    soup = BeautifulSoup(raw, "html.parser")
    tables = soup.select("table.table-public")
    if len(tables) < 2:
        raise ValueError("Project page schema missing required tables")
    main, registration = [table_fields(t) for t in tables[:2]]
    if "[FOR&AGR]" not in main.get("ประเภทโครงการ", ""):
        raise ValueError("Project detail is outside FOR&AGR")
    warnings = []

    def safe_date(value: str | None):
        try:
            return thai_date(value)
        except ValueError as exc:
            warnings.append(str(exc))
            return None

    methodology = main.get("ระเบียนวิธี") or main.get("ระเบียบวิธี")
    expected_key = next((k for k in registration if "ที่คาดว่าจะลดได้" in k), None)
    project = Project(
        project_id=listing["project_id"], registration_number=main.get("เลขที่ขึ้นทะเบียน"),
        program=main.get("โปรแกรม", "UNKNOWN"), group=str(listing["group_id"]),
        name=main.get("ชื่อโครงการ", listing["name"]), methodology=methodology,
        methodology_codes=re.findall(METHOD_PATTERN, methodology or ""),
        status=main.get("สถานภาพโครงการ"), size=main.get("ขนาดโครงการ"),
        location=main.get("ที่ตั้งโครงการ"), developer=main.get("ผู้พัฒนาโครงการ"),
        owner=main.get("เจ้าของโครงการ"), validator=registration.get("ผู้ประเมินภายนอก"),
        registration_date=safe_date(registration.get("วันที่ขึ้นทะเบียน")),
        fiscal_year_be=int(number(registration["ปีงบประมาณ"])) if registration.get("ปีงบประมาณ") else None,
        crediting_start=safe_date(registration.get("วันที่เริ่มคิดเครดิต")),
        crediting_end=safe_date(registration.get("วันที่สิ้นสุดการคิดเครดิต")),
        stated_duration=registration.get("ระยะเวลาการคิดเครดิตของโครงการ"),
        expected_annual_tonnes=number(registration.get(expected_key)) if expected_key else None,
        source_id=source["source_id"], source_url=source["url"],
        raw_fields={"main": main, "registration": registration}, warnings=warnings)
    events, children, documents = [], [], []
    for table_index, table in enumerate(tables):
        fields = table_fields(table)
        if "ครั้งที่" in fields and "วันที่ได้รับการรับรอง" in fields:
            cycle = str(int(number(fields["ครั้งที่"])))
            requested_key = next((k for k in fields if "ที่ขอรับรองในครั้งนี้" in k), None)
            certified = listing["certified_cycles"].get(cycle)
            events.append({"project_id": project.project_id, "cycle": int(cycle),
                           "approval_date": safe_date(fields.get("วันที่ได้รับการรับรอง")),
                           "period_start": safe_date(fields.get("วันที่เริ่มคิดเครดิต")),
                           "period_end": safe_date(fields.get("วันที่สิ้นสุดการคิดเครดิต")),
                           "requested_tonnes": number(fields.get(requested_key)) if requested_key else None,
                           "certified_tonnes": certified,
                           "source_id": source["source_id"], "certified_source_id": listing["source_id"],
                           "source_url": source["url"], "table_index": table_index,
                           "original_values": fields})
        if "เลขที่กลุ่มโครงการย่อย" in fields:
            children.append({"parent_project_id": project.project_id,
                             "subproject_id": fields["เลขที่กลุ่มโครงการย่อย"],
                             "source_id": source["source_id"], "table_index": table_index,
                             "raw_fields": fields})
        for a in table.select("a[href]"):
            href = a["href"]
            if not re.search(r"\.(pdf|docx?|xlsx?)(?:\?|$)", href, re.I):
                continue
            documents.append({"project_id": project.project_id,
                              "url": require_tgo_url(urljoin(BASE, href)),
                              "title": a.get_text(" ", strip=True), "parent_source_id": source["source_id"],
                              "table_index": table_index,
                              "context": "issuance" if "ครั้งที่" in fields else "registration" if table_index == 1 else "other"})
    project.warnings = warnings
    if project.expected_annual_tonnes != listing["expected_annual_tonnes"]:
        project.warnings.append("Listing and registration expected annual tonnes disagree")
    if project.crediting_start and project.crediting_end:
        start, end = date.fromisoformat(project.crediting_start), date.fromisoformat(project.crediting_end)
        if end < start:
            project.warnings.append("Crediting period ends before start")
        stated = re.fullmatch(r"(\d+)\s*ปี", project.stated_duration or "")
        if stated and abs(((end - start).days + 1) / 365.2425 - int(stated[1])) > 0.05:
            project.warnings.append("Stated duration differs from the reported start/end dates")
    return project, events, children, documents


def extract_text(raw: bytes, warning_messages: list[str] | None = None) -> list[tuple[str, str]]:
    """Return locatable passages, retaining page boundaries and avoiding navigation."""
    if raw.startswith(b"%PDF"):
        from pypdf import PdfReader
        class CaptureWarnings(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if warning_messages is not None and message not in warning_messages:
                    warning_messages.append(message)
        logger = logging.getLogger("pypdf")
        handler = CaptureWarnings(level=logging.WARNING)
        previous_propagation = logger.propagate
        logger.addHandler(handler)
        logger.propagate = False
        try:
            reader = PdfReader(io.BytesIO(raw))
            return [(f"page:{i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]
        finally:
            logger.removeHandler(handler)
            logger.propagate = previous_propagation
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.select("script,style,nav,header,footer"):
        tag.decompose()
    tables = soup.select("table.table-public")
    if tables:
        return [(f"table:{i}", t.get_text(" ", strip=True)) for i, t in enumerate(tables)]
    body = soup.find("article") or soup.find("main") or soup.select_one("#sp-main-body") or soup.body or soup
    return [("body", body.get_text(" ", strip=True))]
