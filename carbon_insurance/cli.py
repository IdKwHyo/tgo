from __future__ import annotations

import argparse
import json
import logging
import sys

from .domain import Settings, clean_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TGO-only carbon-absorption insurance research; no invented premiums")
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument("--data-dir", help="Override the data directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "features", "build-index", "train", "evaluate", "assess", "run-all"):
        sub = subparsers.add_parser(name)
        if name in ("ingest", "run-all"):
            sub.add_argument("--offline", action="store_true", help="Replay and integrity-check saved source bytes; do not refresh their observation dates")
        if name == "assess":
            sub.add_argument("--project-id", required=True, help="Project URL ID, e.g. 449 (registration number is a separate identifier)")
        if name == "run-all":
            sub.add_argument("--force-train", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = Settings.load(args.config, args.data_dir)
        if args.command == "ingest":
            from .ingestion import ingest
            result = ingest(settings, offline=args.offline)
        elif args.command == "features":
            from .features import build_features
            result = build_features(settings)
        elif args.command == "build-index":
            from .retrieval import build_index
            result = build_index(settings)
        elif args.command == "train":
            from .modeling import train
            from .pipeline import review_models
            result = train(settings)
            ingestion = json.loads((settings.data_dir / "outputs/ingestion.json").read_text())
            review_models(settings, result, ingestion)
        elif args.command == "evaluate":
            from .pipeline import evaluate
            result = evaluate(settings)
        elif args.command == "assess":
            from .risk import assess
            result = assess(settings, args.project_id)
        else:
            from .pipeline import run_all
            result = run_all(settings, offline=args.offline, force_train=args.force_train)
        print(json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False, default=str))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
