from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .domain import Settings, digest, now, require_tgo_url, write_json
from .ingestion import records
from .parsing import extract_text


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def chunk_id(source_hash: str, locator: str, start: int, text: str) -> str:
    return digest({"source_sha256": source_hash, "locator": locator, "start": start, "text": text})[:28]


def source_program(source: dict) -> str:
    text = (source.get("title", "") + " " + source.get("url", "")).lower()
    if "premium" in text or "t-ver-p-" in text or "/project/2/" in text or "/methodologies/2" in text or "3413-buffer" in text:
        return "Premium T-VER"
    if "standard" in text or "t-ver-s-" in text or "/project/1/" in text or "/methodologies/1" in text:
        return "Standard T-VER"
    return "UNSPECIFIED"


def implementation_fingerprint() -> str:
    return digest({name: digest((Path(__file__).parent / name).read_bytes())
                   for name in ("retrieval.py", "parsing.py")})


def build_index(settings: Settings) -> dict:
    root = settings.data_dir
    ingestion = json.loads((root / "outputs/ingestion.json").read_text())
    sources = sorted(records(root, "knowledge_sources"),
                     key=lambda s: (s["source_id"], s["role"], s.get("project_id") or "", s["title"]))
    project_features = {p["project_id"]: p for p in records(root, "project_features")}
    chunks, skipped, extraction_warnings = [], [], []
    seen = set()
    for source in sources:
        project = project_features.get(source.get("project_id"))
        if project and project["activity_class"] in ("REDUCTION_ONLY", "UNRESOLVED_METHODOLOGY"):
            continue
        require_tgo_url(source["url"])
        raw = (root / source["path"]).read_bytes()
        if digest(raw) != source["sha256"]:
            raise ValueError(f"Raw source integrity check failed: {source['source_id']}")
        try:
            source_warnings = []
            passages = extract_text(raw, source_warnings)
            if source_warnings:
                extraction_warnings.append({"source_id": source["source_id"], "warnings": source_warnings,
                                            "interpretation": "PDF parser repaired or ignored malformed structures; extracted text needs semantic review"})
        except Exception as exc:
            skipped.append({"source_id": source["source_id"], "reason": str(exc)})
            continue
        usable_chars = 0
        for locator, passage in passages:
            text = normalize_text(passage)
            if len(text) < 40:
                continue
            usable_chars += len(text)
            for start in range(0, len(text), 1400):
                excerpt = text[start:start + 1600]
                if len(excerpt) < 40:
                    continue
                identifier = chunk_id(source["sha256"], locator, start, excerpt)
                if identifier in seen:
                    continue
                seen.add(identifier)
                chunks.append({"chunk_id": identifier, "source_id": source["source_id"],
                               "source_url": source["url"], "title": source["title"],
                               "source_date": source.get("last_modified"),
                               "version": re.search(r"version\s+(\d+)", source["title"]).group(1) if re.search(r"version\s+(\d+)", source["title"]) else None,
                               "retrieved_at": source["retrieved_at"], "source_sha256": source["sha256"],
                               "raw_path": source["path"], "locator": locator, "start": start,
                               "text": excerpt, "project_id": source.get("project_id"),
                               "program": source_program(source), "role": source["role"]})
        if usable_chars < 100:
            skipped.append({"source_id": source["source_id"], "reason": "Insufficient extractable text; scanned PDF may require OCR (not silently fabricated)"})
    if len(chunks) < 2:
        raise ValueError("Insufficient official text to build a retrieval index")
    chunks.sort(key=lambda c: c["chunk_id"])
    # Fully local multilingual character features -> latent semantic vectors -> exact cosine search.
    # This is an explicitly labeled alternative local embedding system, not a downloaded transformer.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
                                 max_features=24000, sublinear_tf=True)
    sparse = vectorizer.fit_transform([c["text"] for c in chunks])
    dimensions = min(128, sparse.shape[0] - 1, sparse.shape[1] - 1)
    projector = TruncatedSVD(n_components=dimensions, random_state=19)
    vectors = normalize(projector.fit_transform(sparse)).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir = root / "index"
    faiss.write_index(index, str(index_dir / "tgo.faiss"))
    joblib.dump({"vectorizer": vectorizer, "projector": projector}, index_dir / "embedding.joblib")
    write_json(index_dir / "chunks.json", chunks)
    manifest = {"timestamp": now(), "data_version": ingestion["data_version"],
                "implementation_fingerprint": implementation_fingerprint(),
                "index_version": digest([c["chunk_id"] for c in chunks]), "chunks": len(chunks),
                "sources_indexed": len({c["source_id"] for c in chunks}), "dimensions": dimensions,
                "embedding": "Local character TF-IDF (3–5 grams) plus truncated SVD; FAISS cosine search",
                "retrieval_validation": "Mechanical source/excerpt checks; semantic relevance is not guaranteed by cosine similarity",
                "skipped": skipped, "extraction_warnings": extraction_warnings}
    write_json(index_dir / "manifest.json", manifest)
    write_json(root / "outputs/retrieval_index.json", manifest)
    return manifest


class EvidenceIndex:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "index/manifest.json").read_text())
        ingestion = json.loads((root / "outputs/ingestion.json").read_text())
        if ingestion["data_version"] != self.manifest["data_version"]:
            raise ValueError("RAG index is stale for current data; rebuild-index required")
        self.chunks = json.loads((root / "index/chunks.json").read_text())
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        # Load only locally produced trusted artifacts; never load third-party joblib files.
        self.embedding = joblib.load(root / "index/embedding.joblib")
        self.index = faiss.read_index(str(root / "index/tgo.faiss"))
        self._verified_passages = {}

    def verify(self, passage: dict) -> bool:
        canonical = self.by_id.get(passage.get("chunk_id"))
        if canonical is None or any(passage.get(k) != canonical[k] for k in ("text", "source_url", "source_id")):
            return False
        require_tgo_url(canonical["source_url"])
        raw = (self.root / canonical["raw_path"]).read_bytes()
        if digest(raw) != canonical["source_sha256"]:
            return False
        identifier = chunk_id(canonical["source_sha256"], canonical["locator"], canonical["start"], canonical["text"])
        if identifier != canonical["chunk_id"]:
            return False
        # Independently re-extract and check that the excerpt is present at the declared location.
        if canonical["source_sha256"] not in self._verified_passages:
            self._verified_passages[canonical["source_sha256"]] = dict(extract_text(raw))
        passages = self._verified_passages[canonical["source_sha256"]]
        full_text = normalize_text(passages.get(canonical["locator"], ""))
        return full_text[canonical["start"]:canonical["start"] + len(canonical["text"])] == canonical["text"]

    def search(self, query: str, *, project_id: str | None = None,
               program: str | None = None, top_k: int = 8) -> list[dict]:
        sparse = self.embedding["vectorizer"].transform([query])
        if sparse.nnz == 0:
            return []
        query_vector = normalize(self.embedding["projector"].transform(sparse)).astype("float32")
        scores, ids = self.index.search(query_vector, len(self.chunks))
        ranked = []
        for score, index in zip(scores[0], ids[0]):
            if index < 0 or score <= 0.02:
                continue
            chunk = self.chunks[index]
            if chunk["project_id"] is not None and chunk["project_id"] != project_id:
                continue
            if program and chunk["program"] not in (program, "UNSPECIFIED"):
                continue
            ranked.append({**chunk, "retrieval_score": float(score), "score_semantics": "Cosine similarity, not factual confidence"})
        # Avoid a long PDF crowding out the project's own official registration/issuance tables.
        selected = []
        per_source = {}
        own = [c for c in ranked if c["project_id"] == project_id and c["role"] == "project"]
        general = [c for c in ranked if c["project_id"] is None]
        for candidate in own[:2] + general[:2] + ranked:
            sid = candidate["source_id"]
            if candidate["chunk_id"] in {c["chunk_id"] for c in selected} or per_source.get(sid, 0) >= 3:
                continue
            if not self.verify(candidate):
                raise ValueError("Retrieved evidence failed independent raw-source verification")
            candidate["source_integrity_verified"] = True
            selected.append(candidate)
            per_source[sid] = per_source.get(sid, 0) + 1
            if len(selected) >= top_k:
                break
        return selected
