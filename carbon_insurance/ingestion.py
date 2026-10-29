from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from .domain import Settings, digest, now, write_json
from .parsing import (BASE, listing_parameters, number, parse_csv, parse_listing,
                      parse_method_history, parse_methods, parse_project)
from .store import SourceStore

log = logging.getLogger(__name__)
DATASETS = {
    "trading": ("061c3709-1cdb-4af8-85f6-4c7cc6ce2b51", "1d9ad8ca-70c9-4f6e-bbd5-60eacb09d70e", "cc_02_02.csv"),
    "premium_expected": ("6c14f00b-31b4-49d4-b981-947c10bcab67", "dcb909e1-23c6-417e-9fe0-d6bd0491b116", "pre_02_02.csv"),
    "standard_expected": ("b9f275e5-6bbf-4c3b-907f-3a17c1540259", "bcd5564c-b149-4bb4-814b-55fec10a32b8", "ra_02_02.csv"),
    "standard_issuance": ("90019e9a-228c-4374-bedb-418f761a7a1a", "2ee69181-7498-44f9-9ea6-18ae5292884d", "ra_03_02.csv"),
}
GUIDANCE = {
    "Premium non-permanence requirements": BASE + "/premium-tver/overview/non-permanence-risk",
    "TGO buffer requirements and agriculture exception": "https://ghgreduction.tgo.or.th/en/content_page/item/3413-buffer-credit.html",
    "Premium registration and issuance criteria": "https://ghgreduction.tgo.or.th/en/content_page/item/3410-premium-t-ver-criteria-for-registration.html",
}


def save_table(root: Path, name: str, records: list[dict]) -> None:
    frame = pd.DataFrame(records)
    for column in frame:
        if any(isinstance(v, (dict, list)) for v in frame[column]):
            frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v)
    frame.to_parquet(root / f"processed/{name}.parquet", index=False)
    frame.to_csv(root / f"outputs/{name}.csv", index=False)


def records(root: Path, name: str) -> list[dict]:
    path = root / f"processed/{name}.parquet"
    if not path.exists():
        return []
    result = json.loads(pd.read_parquet(path).to_json(orient="records", force_ascii=False))
    for row in result:
        for key, value in row.items():
            if isinstance(value, str) and value.startswith(("{", "[")):
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
    return result


def ingest(settings: Settings, *, offline: bool = False) -> dict:
    store = SourceStore(settings)
    run_id = "ingest_" + now().replace(":", "").replace(".", "_")
    source_ids: set[str] = set()
    errors, csv_validation, aggregates, metadata, methods, knowledge = [], [], [], [], [], []

    def fetch(url, *, title="", data=None):
        source, raw = store.fetch(url, title=title, data=data, offline=offline)
        source_ids.add(source["source_id"])
        return source, raw

    for kind, (dataset_id, resource_id, filename) in DATASETS.items():
        fallback = f"https://opendata.tgo.or.th/dataset/{dataset_id}/resource/{resource_id}/download/{filename}"
        resource_metadata = {}
        try:
            source, raw = fetch(f"https://opendata.tgo.or.th/api/3/action/package_show?id={dataset_id}", title=f"TGO CKAN metadata: {kind}")
            package = json.loads(raw)
            if not package.get("success"):
                raise ValueError("CKAN returned unsuccessful response")
            resources = package["result"]["resources"]
            matching = [r for r in resources if r["id"] == resource_id]
            if len(matching) != 1:
                raise ValueError("Pinned official CSV resource is no longer in dataset metadata")
            resource_metadata = matching[0]
            metadata.append({"kind": kind, "dataset_id": dataset_id, "resource_id": resource_id,
                             "title": package["result"].get("title"),
                             "metadata_modified": package["result"].get("metadata_modified"),
                             "resource_last_modified": resource_metadata.get("last_modified"),
                             "source_id": source["source_id"]})
        except Exception as exc:
            errors.append({"stage": "dataset_discovery", "source": fallback, "error": str(exc),
                           "action": "Using pinned official CSV URL; discovery metadata unavailable"})
        try:
            source, raw = fetch(resource_metadata.get("url") or fallback, title=f"TGO {kind}")
            parsed, validation = parse_csv(raw, source, kind)
            for row in parsed:
                row["resource_last_modified"] = resource_metadata.get("last_modified")
                entity = f"aggregate:{kind}:{row['fiscal_year_be']}"
                for field, value in row.items():
                    if isinstance(value, (int, float)) and field != "source_row":
                        store.fact(run_id, entity, field, json.dumps(row["original_values"], ensure_ascii=False),
                                   value, "THB/tCO2e" if "price" in field else "THB" if field == "value_baht" else "year" if "year" in field else "tCO2e", source["source_id"], f"csv:row:{row['source_row']}")
            aggregates.extend(parsed)
            csv_validation.append(validation)
        except Exception as exc:
            errors.append({"stage": "csv", "source": fallback, "error": str(exc)})

    listings = []
    listing_counts = []
    for program in (1, 2):
        category_id = None
        for group in (1, 2):
            url = f"{BASE}/database/public/projects/{program}/{group}?lang=th"
            try:
                _, raw = fetch(url, title=f"Category filter: program {program}, group {group}")
                params = listing_parameters(raw, program, group)
                category_id = params["filter_category_id"]
                source, raw = fetch(BASE + "/database/Public/ProjectsResult", data=params,
                                    title=f"FOR&AGR listing: program {program}, group {group}")
                parsed = parse_listing(raw, source)
                listings.extend(parsed)
                listing_counts.append({"program_id": program, "group_id": group,
                                       "internal_category_id": category_id, "rows": len(parsed)})
                for row in parsed:
                    for field, unit in (("expected_annual_tonnes", "tCO2e/year"), ("certified_total_tonnes", "tCO2e")):
                        store.fact(run_id, f"project:{row['project_id']}", field, json.dumps(row["original_values"], ensure_ascii=False),
                                   row[field], unit, source["source_id"], f"listing:row:{row['source_row']}")
                    for cycle, amount in row["certified_cycles"].items():
                        store.fact(run_id, f"event:{row['project_id']}:{cycle}", "certified_tonnes",
                                   row["original_values"][7], amount, "tCO2e", source["source_id"], f"listing:row:{row['source_row']}:cycle:{cycle}")
            except Exception as exc:
                errors.append({"stage": "listing", "source": url, "error": str(exc)})
        if category_id is not None:
            try:
                url = f"{BASE}/database/public/methodologies/{program}?category_id={category_id}&lang=th"
                source, raw = fetch(url, title=f"Category 13 methodology definitions: program {program}")
                definitions = parse_methods(raw, source, program)
                methods.extend(definitions)
                knowledge.append({**source, "role": "methodology_listing", "project_id": None})
                for definition in definitions:
                    if not definition["detail_url"]:
                        continue
                    try:
                        history_source, history_raw = fetch(definition["detail_url"], title=f"{definition['code']} version history: {definition['title']}")
                        methods.extend(parse_method_history(history_raw, history_source, definition))
                        knowledge.append({**history_source, "role": "methodology_history", "project_id": None})
                    except Exception as exc:
                        errors.append({"stage": "methodology_history", "source": definition["detail_url"], "error": str(exc)})
            except Exception as exc:
                errors.append({"stage": "methodologies", "error": str(exc)})

    # The URL ID is distinct from the registration number. Never merge on registration number alone.
    ids = [r["project_id"] for r in listings]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate project URL IDs across listings; investigation required")
    listings.sort(key=lambda r: (r["project_id"] not in settings.sample_project_ids,
                                 r["certified_total_tonnes"] == 0, int(r["project_id"])))
    selected_listings = listings[:settings.max_projects] if settings.max_projects else listings
    projects, events, children, documents = [], [], [], []

    def get_project(listing):
        source, raw = store.fetch(listing["source_url"], title=listing["name"], offline=offline)
        return source, parse_project(raw, source, listing)

    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = {executor.submit(get_project, row): row for row in selected_listings}
        for completed, future in enumerate(as_completed(futures), 1):
            listing = futures[future]
            try:
                source, (project, project_events, subprojects, links) = future.result()
                source_ids.add(source["source_id"])
                value = project.model_dump()
                projects.append(value)
                events.extend(project_events)
                children.extend(subprojects)
                documents.extend(links)
                knowledge.append({**source, "role": "project", "project_id": project.project_id})
                store.fact(run_id, f"project:{project.project_id}", "registration_expected_annual_tonnes",
                           str(project.expected_annual_tonnes), project.expected_annual_tonnes,
                           "tCO2e/year", source["source_id"], "table:1:expected_annual_tonnes")
                if project.fiscal_year_be:
                    store.fact(run_id, f"project:{project.project_id}", "registration_fiscal_year_be",
                               str(project.fiscal_year_be), project.fiscal_year_be, "BE fiscal year",
                               source["source_id"], "table:1:ปีงบประมาณ")
                for event in project_events:
                    store.fact(run_id, f"event:{project.project_id}:{event['cycle']}", "requested_tonnes",
                               json.dumps(event["original_values"], ensure_ascii=False), event["requested_tonnes"],
                               "tCO2e", source["source_id"], f"table:{event['table_index']}:requested_tonnes")
            except Exception as exc:
                errors.append({"stage": "project", "project_id": listing["project_id"],
                               "source": listing["source_url"], "error": str(exc)})
            if completed % 25 == 0 or completed == len(futures):
                log.info("Parsed %s / %s project pages", completed, len(futures))

    # Guidance is a separate evidence population, never an insurance claims table.
    guidance_pdf_links = []
    for title, url in GUIDANCE.items():
        try:
            source, raw = fetch(url, title=title)
            knowledge.append({**source, "role": "guidance", "project_id": None})
            if "non-permanence" in url:
                soup = BeautifulSoup(raw, "html.parser")
                guidance_pdf_links.extend({"url": urljoin(url, a["href"]), "title": a.get_text(" ", strip=True),
                                           "role": "guidance_pdf", "project_id": None}
                                          for a in soup.select("a[href]") if ".pdf" in a["href"].lower())
        except Exception as exc:
            errors.append({"stage": "guidance", "source": url, "error": str(exc)})

    # Fetch Category 13 methodology PDFs, then a bounded reproducible sample of project documents.
    pdf_queue = list(guidance_pdf_links)
    for method in methods:
        if method.get("is_historical"):
            continue
        for link in method["links"]:
            pdf_queue.append({"url": link, "title": f"{method['code']} version {method['version']}: {method['title']}",
                              "role": "methodology_pdf", "project_id": None})
    project_by_id = {p["project_id"]: p for p in projects}
    # Prefer sample projects and monitoring/verification reports; preserve all other URLs in the inventory.
    def priority(doc):
        return (doc["project_id"] not in settings.sample_project_ids,
                not any(word in doc["title"] for word in ("ทวนสอบ", "ติดตาม", "ข้อเสนอ")),
                int(doc["project_id"]), doc["url"])
    picked = []
    for doc in sorted(documents, key=priority):
        # Avoid unrelated reduction-only project PDFs even though they are in Category 13.
        codes = project_by_id[doc["project_id"]]["methodology_codes"]
        if any(code in {"T-VER-S-METH-13-05", "T-VER-P-METH-13-08"} for code in codes):
            continue
        if doc["url"].lower().endswith(".pdf") and doc["url"] not in {d["url"] for d in picked}:
            picked.append(doc)
        if len(picked) >= settings.max_project_pdfs:
            break
    if settings.max_project_pdfs == 0:
        picked = []
    pdf_queue.extend({**d, "role": "project_pdf"} for d in picked)
    for pdf in pdf_queue:
        try:
            source, raw = fetch(pdf["url"], title=pdf["title"])
            if not raw.startswith(b"%PDF"):
                raise ValueError("PDF link returned non-PDF content")
            knowledge.append({**source, "role": pdf["role"], "project_id": pdf["project_id"]})
        except Exception as exc:
            errors.append({"stage": "pdf", "source": pdf["url"], "error": str(exc)})

    projects.sort(key=lambda p: int(p["project_id"]))
    events.sort(key=lambda e: (int(e["project_id"]), e["cycle"]))
    children.sort(key=lambda c: (int(c["parent_project_id"]), c["subproject_id"]))
    documents.sort(key=lambda d: (int(d["project_id"]), d["table_index"], d["url"]))
    knowledge.sort(key=lambda k: (k["source_id"], k["role"], k.get("project_id") or ""))
    for name, rows in {"aggregates": aggregates, "projects": projects, "issuance_events": events,
                       "subprojects": children, "documents": documents, "methodologies": methods,
                       "listings": listings, "knowledge_sources": knowledge, "dataset_metadata": metadata}.items():
        save_table(store.root, name, rows)
    semantic_version = digest({"aggregates": aggregates, "projects": sorted(projects, key=lambda p: int(p["project_id"])),
                               "events": sorted(events, key=lambda e: (int(e["project_id"]), e["cycle"])),
                               "source_hashes": sorted((s["request_key"], s["sha256"]) for s in store.inventory(list(source_ids)))})
    summary = {"run_id": run_id, "timestamp": now(), "data_version": semantic_version,
               "offline_replay": offline, "csv_validation": csv_validation, "listing_counts": listing_counts,
               "discovered_project_count": len(listings), "parsed_project_count": len(projects),
               "project_limit_applied": settings.max_projects is not None,
               "issuance_events": len(events), "subprojects": len(children),
               "document_links": len(documents), "downloaded_project_pdfs": sum(k["role"] == "project_pdf" for k in knowledge),
               "source_ids": sorted(source_ids), "errors": errors,
               "scope_warnings": ["Category 13 market/annual aggregates include reduction activities; no removal-only transaction benchmark is published in these files.",
                                  "Fiscal years are Oct 1–Sep 30; CSVs can be year-to-date snapshots even after that fiscal year has ended.",
                                  "Registration, issuance and trading aggregates are different populations and are never divided to estimate delivery performance.",
                                  "Linked-document download coverage is bounded by configuration; an unextracted disclosure is missing evidence, not zero risk."]}
    write_json(store.root / "outputs/ingestion.json", summary)
    with store.connect() as db:
        db.execute("INSERT INTO runs VALUES(?,?,?,?)", (run_id, summary["timestamp"], "ingest", json.dumps(summary)))
    store.export(run_id, sorted(source_ids))
    log.info("Ingestion complete: %s projects, %s events, %s errors", len(projects), len(events), len(errors))
    return summary
