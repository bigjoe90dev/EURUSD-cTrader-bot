from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import dump_json, dump_yaml, list_files, load_yaml, parse_utc_ts, sha256_file, to_iso_z
from .validate_dataset import validate_file


def _dataset_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    eur = config.get("eurusd", {})
    btc = config.get("btcusdt", {})
    xau = config.get("xauusd", {})

    return [
        {
            "id": "eurusd",
            "source": "local_sqlite",
            "symbol": eur.get("symbol", "EURUSD"),
            "timeframe": eur.get("timeframe", "1m"),
            "kind": "ohlc",
            "path": eur.get("canonical_1m_path", "research/data_cache/eurusd/normalized/eurusd_ohlc_1m.parquet"),
            "files_root": eur.get("files_root", "research/data_cache/eurusd"),
            "required_start_ts_utc": eur.get("required_start_ts_utc"),
            "required_end_ts_utc": eur.get("required_end_ts_utc"),
        },
        {
            "id": "btcusdt",
            "source": "binance",
            "symbol": btc.get("symbol", "BTCUSDT"),
            "timeframe": btc.get("timeframe", "1m"),
            "kind": "ohlc",
            "path": btc.get("canonical_1m_path", "research/data_cache/btcusdt/binance/normalized/btcusdt_1m.parquet"),
            "files_root": btc.get("files_root", "research/data_cache/btcusdt/binance"),
            "required_start_ts_utc": btc.get("required_start_ts_utc"),
            "required_end_ts_utc": btc.get("required_end_ts_utc"),
        },
        {
            "id": "xauusd",
            "source": "dukascopy",
            "symbol": xau.get("symbol", "XAUUSD"),
            "timeframe": xau.get("timeframe", "1m"),
            "kind": "ohlc",
            "path": xau.get("canonical_1m_path", "research/data_cache/xauusd/dukascopy/normalized/xauusd_1m.parquet"),
            "files_root": xau.get("files_root", "research/data_cache/xauusd/dukascopy"),
            "required_start_ts_utc": xau.get("required_start_ts_utc"),
            "required_end_ts_utc": xau.get("required_end_ts_utc"),
        },
    ]


def _check_coverage(spec: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    actual_start = validation.get("start_ts_utc")
    actual_end = validation.get("end_ts_utc")
    req_start = spec.get("required_start_ts_utc")
    req_end = spec.get("required_end_ts_utc")

    if req_start and actual_start:
        if parse_utc_ts(actual_start) > parse_utc_ts(req_start):
            errs.append(
                f"coverage start too late: actual={actual_start}, required<={to_iso_z(parse_utc_ts(req_start))}"
            )
    if req_end and actual_end:
        if parse_utc_ts(actual_end) < parse_utc_ts(req_end):
            errs.append(
                f"coverage end too early: actual={actual_end}, required>={to_iso_z(parse_utc_ts(req_end))}"
            )
    return errs


def build_manifest(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    validation_details: dict[str, Any] = {}

    for spec in _dataset_specs(config):
        path = Path(spec["path"])
        if not path.exists():
            errors.append(f"{spec['id']}: dataset file not found: {path}")
            continue

        validation = validate_file(path, kind=spec["kind"], expected_seconds=60)
        validation_details[spec["id"]] = validation

        if not validation["ok"]:
            for err in validation["errors"]:
                errors.append(f"{spec['id']}: {err}")

        cov_errs = _check_coverage(spec, validation)
        if cov_errs:
            for err in cov_errs:
                errors.append(f"{spec['id']}: {err}")

        entry = {
            "dataset_id": spec["id"],
            "source": spec["source"],
            "symbol": spec["symbol"],
            "timeframe": spec["timeframe"],
            "start_ts_utc": validation.get("start_ts_utc"),
            "end_ts_utc": validation.get("end_ts_utc"),
            "row_count": validation.get("row_count"),
            "gaps_found": validation.get("gaps_found"),
            "longest_gap_seconds": validation.get("longest_gap_seconds"),
            "duplicates_found": validation.get("duplicates_found"),
            "checksum_sha256": sha256_file(path),
            "files_on_disk": list_files(spec["files_root"]),
            "canonical_file": str(path),
            "required_start_ts_utc": spec.get("required_start_ts_utc"),
            "required_end_ts_utc": spec.get("required_end_ts_utc"),
        }
        entries.append(entry)

    manifest = {
        "ok": len(errors) == 0,
        "datasets": entries,
        "errors": errors,
    }
    return manifest, validation_details


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset manifest with strict validation checks")
    parser.add_argument("--config", default="research/config_data.yml", help="Config YAML path")
    parser.add_argument(
        "--out",
        help="Manifest output path (defaults to manifest.path in config or research/data_manifest.yml)",
    )
    parser.add_argument(
        "--validation-out",
        default="research/artefacts/logs/data_validation_report.json",
        help="Detailed validation report JSON path",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    manifest, validation_details = build_manifest(config)

    out_path = args.out or config.get("manifest", {}).get("path", "research/data_manifest.yml")
    dump_yaml(out_path, manifest)
    dump_json(args.validation_out, validation_details)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {out_path}")
    print(f"Wrote validation details to {args.validation_out}")

    if not manifest["ok"]:
        raise SystemExit("Validation/coverage checks failed; manifest marked not ok.")


if __name__ == "__main__":
    main()
