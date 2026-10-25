#!/usr/bin/env python3
"""Track tropical-cyclone centres in AI-model GRIB forecasts.

The tracker is deliberately model-independent.  It uses the same fields,
candidate generation, motion constraint and score for FourCastNet,
Pangu-Weather and GraphCast.  Results are written to versioned CSV/JSON files
so that a run can be reproduced instead of existing only in a terminal log.

Default common-variable method:
  * candidate centres: low MSLP grid points near the motion-predicted centre;
  * tie breakers: distance, pressure deficit, and local maximum 10 m wind;
  * 850 hPa vorticity support is implemented but disabled by default so all
    three models are compared using an identical common-variable set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from eccodes import (
        codes_get,
        codes_get_array,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )
except ImportError:  # pragma: no cover - depends on server environment
    codes_get = None
    codes_get_array = None
    codes_get_values = None
    codes_grib_new_from_file = None
    codes_release = None


UTC = timezone.utc
EARTH_RADIUS_KM = 6371.0088
MSLP_NAMES = {"msl", "prmsl"}
U10_NAMES = {"10u", "u10"}
V10_NAMES = {"10v", "v10"}
ISOBARIC_TYPES = {"isobaricInhPa", "isobaricInPa"}


@dataclass
class GridFields:
    lat: np.ndarray
    lon: np.ndarray
    msl_hpa: np.ndarray
    u10_ms: np.ndarray | None = None
    v10_ms: np.ndarray | None = None
    u850_ms: np.ndarray | None = None
    v850_ms: np.ndarray | None = None


@dataclass
class Candidate:
    lat: float
    lon: float
    row: int
    col: int
    msl_hpa: float
    distance_km: float
    pressure_deficit_hpa: float
    max_wind_ms: float | None
    vorticity_s1: float | None
    score: float


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M UTC").replace(tzinfo=UTC)


def utc_now_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def normalise_lon(lon: float | np.ndarray) -> float | np.ndarray:
    return np.mod(lon, 360.0)


def signed_lon_delta(lon_to: float, lon_from: float) -> float:
    return (lon_to - lon_from + 180.0) % 360.0 - 180.0


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad((np.asarray(lon2) - np.asarray(lon1) + 180.0) % 360.0 - 180.0)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def safe_get(gid: int, key: str, default: Any = None) -> Any:
    try:
        return codes_get(gid, key)
    except Exception:
        return default


def reshape_message(gid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ni = int(codes_get(gid, "Ni"))
    nj = int(codes_get(gid, "Nj"))
    values = np.asarray(codes_get_values(gid), dtype=np.float64)
    lats = np.asarray(codes_get_array(gid, "latitudes"), dtype=np.float64)
    lons = np.asarray(codes_get_array(gid, "longitudes"), dtype=np.float64)
    expected = ni * nj
    if values.size != expected or lats.size != expected or lons.size != expected:
        raise ValueError(
            f"只支持规则全球网格：Ni={ni}, Nj={nj}, values={values.size}"
        )
    return (
        values.reshape(nj, ni),
        lats.reshape(nj, ni),
        normalise_lon(lons.reshape(nj, ni)),
    )


def is_level_850(gid: int) -> bool:
    level_type = str(safe_get(gid, "typeOfLevel", ""))
    level = float(safe_get(gid, "level", math.nan))
    if level_type == "isobaricInPa":
        level /= 100.0
    return level_type in ISOBARIC_TYPES and math.isclose(level, 850.0)


def read_grib_fields(path: Path) -> GridFields:
    if codes_grib_new_from_file is None:
        raise RuntimeError(
            "缺少 Python eccodes。请在已安装 ai-models 的 graphcast 环境中运行本脚本。"
        )
    selected: dict[str, np.ndarray] = {}
    grid_lat: np.ndarray | None = None
    grid_lon: np.ndarray | None = None

    with path.open("rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                short_name = str(safe_get(gid, "shortName", ""))
                key: str | None = None
                if short_name in MSLP_NAMES and "msl" not in selected:
                    key = "msl"
                elif short_name in U10_NAMES and "u10" not in selected:
                    key = "u10"
                elif short_name in V10_NAMES and "v10" not in selected:
                    key = "v10"
                elif short_name == "u" and is_level_850(gid) and "u850" not in selected:
                    key = "u850"
                elif short_name == "v" and is_level_850(gid) and "v850" not in selected:
                    key = "v850"

                if key is not None:
                    values, lats, lons = reshape_message(gid)
                    if grid_lat is None:
                        grid_lat, grid_lon = lats, lons
                    elif values.shape != grid_lat.shape:
                        raise ValueError(f"{path} 内所需字段网格形状不一致")
                    selected[key] = values
            finally:
                codes_release(gid)

    if "msl" not in selected or grid_lat is None or grid_lon is None:
        raise ValueError(f"{path} 中找不到海平面气压 msl")

    msl = selected["msl"]
    finite = msl[np.isfinite(msl)]
    if finite.size == 0:
        raise ValueError(f"{path} 的 msl 全部为空")
    # GRIB normally stores MSLP in Pa.  This also tolerates an already-hPa file.
    msl_hpa = msl / 100.0 if float(np.nanmedian(finite)) > 2000.0 else msl
    return GridFields(
        lat=grid_lat,
        lon=grid_lon,
        msl_hpa=msl_hpa,
        u10_ms=selected.get("u10"),
        v10_ms=selected.get("v10"),
        u850_ms=selected.get("u850"),
        v850_ms=selected.get("v850"),
    )


def motion_anchor(history: list[tuple[float, float]], factor: float) -> tuple[float, float]:
    if len(history) < 2:
        return history[-1]
    prev_lat, prev_lon = history[-2]
    last_lat, last_lon = history[-1]
    lat = float(np.clip(last_lat + factor * (last_lat - prev_lat), -89.5, 89.5))
    lon = float(normalise_lon(last_lon + factor * signed_lon_delta(last_lon, prev_lon)))
    return lat, lon


def relative_vorticity_at(fields: GridFields, row: int, col: int) -> float | None:
    if fields.u850_ms is None or fields.v850_ms is None:
        return None
    nj, ni = fields.lat.shape
    if row <= 0 or row >= nj - 1:
        return None
    cm = (col - 1) % ni
    cp = (col + 1) % ni
    rm, rp = row - 1, row + 1

    dx_km = float(
        haversine_km(
            fields.lat[row, cm], fields.lon[row, cm],
            fields.lat[row, cp], fields.lon[row, cp],
        )
    )
    dy_km = float(
        haversine_km(
            fields.lat[rm, col], fields.lon[rm, col],
            fields.lat[rp, col], fields.lon[rp, col],
        )
    )
    if dx_km <= 0.0 or dy_km <= 0.0:
        return None

    lon_sign = math.copysign(1.0, signed_lon_delta(fields.lon[row, cp], fields.lon[row, cm]))
    lat_sign = math.copysign(1.0, fields.lat[rp, col] - fields.lat[rm, col])
    dv_dx = (fields.v850_ms[row, cp] - fields.v850_ms[row, cm]) / (dx_km * 1000.0 * lon_sign)
    du_dy = (fields.u850_ms[rp, col] - fields.u850_ms[rm, col]) / (dy_km * 1000.0 * lat_sign)
    value = float(dv_dx - du_dy)
    return value if math.isfinite(value) else None


def find_candidates(
    fields: GridFields,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
    cfg: dict[str, float],
) -> list[Candidate]:
    distance = np.asarray(
        haversine_km(anchor_lat, anchor_lon, fields.lat, fields.lon),
        dtype=np.float64,
    )
    mask = (distance <= radius_km) & np.isfinite(fields.msl_hpa)
    flat = np.flatnonzero(mask)
    if flat.size == 0:
        return []

    pressure = fields.msl_hpa.ravel()[flat]
    # Inspect more low-pressure cells than the final candidate count, then
    # enforce physical separation so one depression does not fill the list.
    inspect_count = min(flat.size, max(int(cfg["maximum_candidates"]) * 100, 500))
    # Explicit total order: pressure first, native GRIB flat grid index second.
    # Avoid implementation-dependent ties in argpartition/default quicksort.
    order = np.lexsort((flat, pressure))[:inspect_count]

    regional_median = float(np.nanmedian(fields.msl_hpa[mask]))
    wind_speed = None
    if fields.u10_ms is not None and fields.v10_ms is not None:
        wind_speed = np.hypot(fields.u10_ms, fields.v10_ms)

    selected: list[Candidate] = []
    for position in order:
        index = int(flat[position])
        row, col = np.unravel_index(index, fields.msl_hpa.shape)
        lat = float(fields.lat[row, col])
        lon = float(fields.lon[row, col])
        if any(
            float(haversine_km(lat, lon, item.lat, item.lon))
            < float(cfg["candidate_separation_km"])
            for item in selected
        ):
            continue

        centre_pressure = float(fields.msl_hpa[row, col])
        centre_distance = float(distance[row, col])
        deficit = max(0.0, regional_median - centre_pressure)
        max_wind: float | None = None
        if wind_speed is not None:
            local_distance = np.asarray(haversine_km(lat, lon, fields.lat, fields.lon))
            local = (local_distance <= float(cfg["wind_radius_km"])) & np.isfinite(wind_speed)
            if np.any(local):
                max_wind = float(np.nanmax(wind_speed[local]))

        vorticity = relative_vorticity_at(fields, row, col)
        hemisphere_vorticity = None
        if vorticity is not None:
            hemisphere_vorticity = vorticity if lat >= 0 else -vorticity

        score = float(cfg["distance_weight"]) * centre_distance / radius_km
        score -= float(cfg["pressure_deficit_weight"]) * deficit / float(cfg["pressure_scale_hpa"])
        if max_wind is not None:
            score -= float(cfg["wind_weight"]) * max_wind / float(cfg["wind_scale_ms"])
        if hemisphere_vorticity is not None:
            score -= (
                float(cfg["vorticity_weight"])
                * max(0.0, hemisphere_vorticity)
                / float(cfg["vorticity_scale_s-1"])
            )

        selected.append(
            Candidate(
                lat=lat,
                lon=lon,
                row=int(row),
                col=int(col),
                msl_hpa=centre_pressure,
                distance_km=centre_distance,
                pressure_deficit_hpa=deficit,
                max_wind_ms=max_wind,
                vorticity_s1=vorticity,
                score=score,
            )
        )
        if len(selected) >= int(cfg["maximum_candidates"]):
            break

    return sorted(selected, key=lambda item: item.score)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def output_path(
    root: Path,
    model_cfg: dict[str, str],
    job_id: str,
    lead: int,
) -> Path:
    return (
        root
        / model_cfg["forecast_root"]
        / job_id
        / model_cfg["file_template"].format(step=lead)
    )


def optional_round(value: float | None, digits: int) -> str | float:
    return "" if value is None or not math.isfinite(value) else round(value, digits)


def summarise(track_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in track_rows:
        grouped.setdefault((str(row["model"]), str(row["case_id"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (model, case_id), rows in sorted(grouped.items()):
        valid = [r for r in rows if r["status"] in {"INITIAL_ANCHOR", "TRACKED"}]
        forecast = [r for r in valid if int(r["lead_hours"]) > 0]
        errors = [float(r["track_error_km"]) for r in forecast if r["track_error_km"] != ""]
        result.append(
            {
                "model": model,
                "case_id": case_id,
                "TC_ID": rows[0]["TC_ID"],
                "name": rows[0]["name"],
                "expected_points": len(rows),
                "tracked_points": len(valid),
                "forecast_points": len(forecast),
                "mean_track_error_km": optional_round(float(np.mean(errors)), 2) if errors else "",
                "max_track_error_km": optional_round(max(errors), 2) if errors else "",
                "last_lead_hours": max((int(r["lead_hours"]) for r in valid), default=-1),
                "complete": len(valid) == len(rows),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--models", default="all", help="all or comma-separated model names")
    parser.add_argument("--case-id")
    parser.add_argument("--job-id")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    all_models: dict[str, dict[str, str]] = config["models"]
    if args.models == "all":
        model_names = list(all_models)
    else:
        model_names = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = sorted(set(model_names) - set(all_models))
        if unknown:
            raise SystemExit(f"配置中没有这些模型：{', '.join(unknown)}")

    cases = load_csv(args.cases)
    if args.case_id:
        cases = [row for row in cases if row["case_id"] == args.case_id]
    if args.job_id:
        cases = [row for row in cases if row["forecast_job_id"] == args.job_id]
    if not cases:
        raise SystemExit("没有符合筛选条件的 case")

    truth_rows = load_csv(args.truth)
    truth: dict[tuple[str, int], dict[str, str]] = {
        (row["case_id"], int(row["lead_hours"])): row for row in truth_rows
    }
    run_id = args.run_id or utc_now_id()
    output_root = (args.output_root or root / "results" / "tracks" / f"run_{run_id}").resolve()

    if args.dry_run:
        print(f"models={model_names}")
        print(f"cases={len(cases)}")
        print(f"output_root={output_root}")
        for model in model_names:
            first = cases[0]
            print(
                model,
                output_path(root, all_models[model], first["forecast_job_id"], 6),
            )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    cfg = config["tracking"]
    track_rows: list[dict[str, Any]] = []
    failures = 0

    for model in model_names:
        model_cfg = all_models[model]
        for case_index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            job_id = case["forecast_job_id"]
            lead_max = int(case["forecast_hours"])
            initial_truth = truth.get((case_id, 0))
            if initial_truth is None:
                raise SystemExit(f"真值表缺少 {case_id} lead=0")

            initial_lat = float(initial_truth["jma_lat_deg_n"])
            initial_lon = float(normalise_lon(float(initial_truth["jma_lon_deg_e"])))
            history = [(initial_lat, initial_lon)]
            print(f"[{model} {case_index}/{len(cases)}] {case_id}")

            base = {
                "model": model,
                "case_id": case_id,
                "TC_ID": case["TC_ID"],
                "name": case["name"],
                "forecast_job_id": job_id,
            }
            track_rows.append(
                {
                    **base,
                    "lead_hours": 0,
                    "valid_utc": initial_truth["valid_utc"],
                    "pred_lat_deg_n": initial_lat,
                    "pred_lon_deg_e": initial_lon,
                    "central_msl_hpa": initial_truth["jma_central_pressure_hpa"],
                    "max_10m_wind_ms": "",
                    "anchor_lat_deg_n": initial_lat,
                    "anchor_lon_deg_e": initial_lon,
                    "candidate_distance_km": 0.0,
                    "pressure_deficit_hpa": "",
                    "vorticity_850_s-1": "",
                    "candidate_score": "",
                    "candidate_count": "",
                    "search_radius_km": 0.0,
                    "jma_lat_deg_n": initial_lat,
                    "jma_lon_deg_e": initial_lon,
                    "track_error_km": 0.0,
                    "status": "INITIAL_ANCHOR",
                    "source_file": "JMA truth lead 0",
                }
            )

            for lead in range(6, lead_max + 1, 6):
                valid = truth.get((case_id, lead))
                valid_utc = (
                    valid["valid_utc"]
                    if valid
                    else (parse_utc(case["init_utc"]) + timedelta(hours=lead)).strftime(
                        "%Y-%m-%d %H:%M UTC"
                    )
                )
                path = output_path(root, model_cfg, job_id, lead)
                anchor_lat, anchor_lon = motion_anchor(
                    history, float(cfg["motion_extrapolation_factor"])
                )
                radius = min(
                    float(cfg["maximum_search_radius_km"]),
                    float(cfg["base_search_radius_km"])
                    + (lead // 6 - 1) * float(cfg["search_radius_growth_km_per_step"]),
                )
                row: dict[str, Any] = {
                    **base,
                    "lead_hours": lead,
                    "valid_utc": valid_utc,
                    "anchor_lat_deg_n": round(anchor_lat, 4),
                    "anchor_lon_deg_e": round(anchor_lon, 4),
                    "search_radius_km": round(radius, 1),
                    "source_file": str(path),
                }
                if not path.exists():
                    row["status"] = "MISSING_FORECAST_FILE"
                    track_rows.append(row)
                    failures += 1
                    break
                try:
                    fields = read_grib_fields(path)
                    candidates = find_candidates(fields, anchor_lat, anchor_lon, radius, cfg)
                    if not candidates:
                        raise ValueError("搜索范围内没有有效候选点")
                    best = candidates[0]
                except Exception as exc:
                    row["status"] = "TRACKING_ERROR"
                    row["error"] = str(exc)
                    track_rows.append(row)
                    failures += 1
                    break

                history.append((best.lat, best.lon))
                row.update(
                    {
                        "pred_lat_deg_n": round(best.lat, 4),
                        "pred_lon_deg_e": round(best.lon, 4),
                        "central_msl_hpa": round(best.msl_hpa, 2),
                        "max_10m_wind_ms": optional_round(best.max_wind_ms, 3),
                        "candidate_distance_km": round(best.distance_km, 2),
                        "pressure_deficit_hpa": round(best.pressure_deficit_hpa, 2),
                        "vorticity_850_s-1": optional_round(best.vorticity_s1, 8),
                        "candidate_score": round(best.score, 6),
                        "candidate_count": len(candidates),
                        "status": "TRACKED",
                    }
                )
                if valid:
                    jma_lat = float(valid["jma_lat_deg_n"])
                    jma_lon = float(normalise_lon(float(valid["jma_lon_deg_e"])))
                    row["jma_lat_deg_n"] = jma_lat
                    row["jma_lon_deg_e"] = jma_lon
                    row["track_error_km"] = round(
                        float(haversine_km(best.lat, best.lon, jma_lat, jma_lon)), 2
                    )
                track_rows.append(row)

    track_fields = [
        "model", "case_id", "TC_ID", "name", "forecast_job_id",
        "lead_hours", "valid_utc", "pred_lat_deg_n", "pred_lon_deg_e",
        "central_msl_hpa", "max_10m_wind_ms", "anchor_lat_deg_n",
        "anchor_lon_deg_e", "candidate_distance_km", "pressure_deficit_hpa",
        "vorticity_850_s-1", "candidate_score", "candidate_count",
        "search_radius_km", "jma_lat_deg_n", "jma_lon_deg_e",
        "track_error_km", "status", "source_file", "error",
    ]
    write_csv(output_root / "track_points.csv", track_rows, track_fields)
    summaries = summarise(track_rows)
    summary_fields = [
        "model", "case_id", "TC_ID", "name", "expected_points",
        "tracked_points", "forecast_points", "mean_track_error_km",
        "max_track_error_km", "last_lead_hours", "complete",
    ]
    write_csv(output_root / "case_summary.csv", summaries, summary_fields)

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "project_root": str(root),
        "models": model_names,
        "case_count": len(cases),
        "track_row_count": len(track_rows),
        "failure_count": failures,
        "config": config,
        "arguments": vars(args) | {
            "project_root": str(args.project_root),
            "config": str(args.config),
            "cases": str(args.cases),
            "truth": str(args.truth),
            "output_root": str(output_root),
        },
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果目录：{output_root}")
    print(f"追踪记录：{len(track_rows)}，失败：{failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
