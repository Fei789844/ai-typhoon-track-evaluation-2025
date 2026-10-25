#!/usr/bin/env python3
"""Generate the auditable 2025 WNP 120 h case catalogue from JMA best track.

Protocol
--------
1. Select JMA storms with international IDs 2501--2527.
2. Define first TS as the first JMA point with grade 3, 4, or 5.
3. Choose the first 06 or 18 UTC cycle not earlier than first TS.
4. Add later initialization cases every 12 hours.
5. Retain a case only when all 21 six-hour truth points from lead 0 to 120
   exist and remain tropical (JMA grade 2, 3, 4, or 5).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


UTC = timezone.utc
TS_OR_HIGHER = {3, 4, 5}
TROPICAL_GRADES = {2, 3, 4, 5}
GRADE_LABELS = {
    2: "TD",
    3: "TS",
    4: "STS",
    5: "TY",
    6: "EXTRATROPICAL",
    7: "ENTERING_RSMC_AREA",
    9: "TS_OR_HIGHER",
}
SOURCE_URL = (
    "https://www.jma.go.jp/jma/jma-eng/jma-center/"
    "rsmc-hp-pub-eg/Besttracks/bst_all.zip"
)


@dataclass(frozen=True)
class Point:
    tc_id: str
    name: str
    time: datetime
    grade: int
    lat: float
    lon: float
    central_pressure_hpa: int | None
    max_wind_kt: int | None
    source_line: str


def parse_optional_int(text: str) -> int | None:
    value = text.strip()
    return int(value) if value else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_to_0618(time: datetime) -> datetime:
    """Return the first 06/18 UTC cycle that is not earlier than time."""
    if time.hour <= 6:
        return time.replace(hour=6, minute=0, second=0, microsecond=0)
    if time.hour <= 18:
        return time.replace(hour=18, minute=0, second=0, microsecond=0)
    return (time + timedelta(days=1)).replace(
        hour=6, minute=0, second=0, microsecond=0
    )


def parse_jma_zip(path: Path) -> tuple[dict[str, list[Point]], dict[str, dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("bst_all.txt")
        text = archive.read("bst_all.txt").decode("ascii", errors="strict")

    tracks: dict[str, list[Point]] = {}
    headers: dict[str, dict[str, str]] = {}
    current_id: str | None = None
    current_name = ""

    for line in text.splitlines():
        if line.startswith("66666"):
            tc_id = line[6:10].strip()
            current_id = tc_id if tc_id.startswith("25") else None
            current_name = line[30:50].strip()
            if current_id is not None:
                headers[current_id] = {
                    "TC_ID": current_id,
                    "name": current_name,
                    "line_count_declared": line[12:15].strip(),
                    "termination_flag": line[26:27].strip(),
                    "hours_to_final_analysis": line[28:29].strip(),
                    "latest_revision": line[64:72].strip(),
                }
                tracks[current_id] = []
            continue

        if current_id is None or len(line) < 36 or not line[:8].isdigit():
            continue

        tracks[current_id].append(
            Point(
                tc_id=current_id,
                name=current_name,
                time=datetime.strptime(line[:8], "%y%m%d%H").replace(tzinfo=UTC),
                grade=int(line[13]),
                lat=int(line[15:18]) / 10,
                lon=int(line[19:23]) / 10,
                central_pressure_hpa=parse_optional_int(line[24:28]),
                max_wind_kt=parse_optional_int(line[33:36]),
                source_line=line.rstrip(),
            )
        )

    for points in tracks.values():
        points.sort(key=lambda point: point.time)

    expected = {f"25{i:02d}" for i in range(1, 28)}
    if set(tracks) != expected:
        missing = sorted(expected - set(tracks))
        extra = sorted(set(tracks) - expected)
        raise ValueError(f"Expected 2501--2527; missing={missing}, extra={extra}")

    archive_meta = {
        "member_name": info.filename,
        "member_size_bytes": str(info.file_size),
        "member_modified": "%04d-%02d-%02d %02d:%02d:%02d"
        % info.date_time,
    }
    headers["_archive"] = archive_meta
    return tracks, headers


def iso_hour(time: datetime) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jma-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks, headers = parse_jma_zip(args.jma_zip)

    storm_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []

    for tc_id in sorted(tracks):
        points = tracks[tc_id]
        header = headers[tc_id]
        first_ts_point = next(point for point in points if point.grade in TS_OR_HIGHER)
        first_init = ceil_to_0618(first_ts_point.time)
        tropical_points = [point for point in points if point.grade in TROPICAL_GRADES]
        tropical_by_time = {point.time: point for point in tropical_points}
        last_tropical = max(tropical_by_time)

        case_inits: list[datetime] = []
        candidate = first_init
        while candidate + timedelta(hours=120) <= last_tropical:
            required = [candidate + timedelta(hours=lead) for lead in range(0, 121, 6)]
            if all(time in tropical_by_time for time in required):
                case_inits.append(candidate)
            candidate += timedelta(hours=12)

        remaining_hours = int((last_tropical - first_init).total_seconds() // 3600)
        storm_rows.append(
            {
                "TC_ID": tc_id,
                "name": first_ts_point.name,
                "jma_latest_revision": header["latest_revision"],
                "first_jma_record_utc": iso_hour(points[0].time),
                "first_ts_utc": iso_hour(first_ts_point.time),
                "first_ts_grade": GRADE_LABELS[first_ts_point.grade],
                "first_eligible_init_utc": iso_hour(first_init),
                "last_tropical_utc": iso_hour(last_tropical),
                "hours_available_from_first_init": remaining_hours,
                "complete_120h_case_count": len(case_inits),
                "eligibility": (
                    "MULTIPLE_CASES"
                    if len(case_inits) > 1
                    else "ONE_CASE"
                    if len(case_inits) == 1
                    else "INSUFFICIENT_LIFETIME"
                ),
                "first_case_init_utc": iso_hour(case_inits[0]) if case_inits else "",
                "last_case_init_utc": iso_hour(case_inits[-1]) if case_inits else "",
                "shortfall_to_120h": max(0, 120 - remaining_hours),
            }
        )

        for case_seq, init in enumerate(case_inits, start=1):
            end = init + timedelta(hours=120)
            init_point = tropical_by_time[init]
            end_point = tropical_by_time[end]
            case_id = f"{tc_id}_{first_ts_point.name}_{init:%Y%m%d_%H}00_120h"
            case_rows.append(
                {
                    "case_id": case_id,
                    "TC_ID": tc_id,
                    "name": first_ts_point.name,
                    "case_seq_within_storm": case_seq,
                    "first_ts_utc": iso_hour(first_ts_point.time),
                    "first_eligible_init_utc": iso_hour(first_init),
                    "init_utc": iso_hour(init),
                    "end_utc": iso_hour(end),
                    "forecast_hours": 120,
                    "rolling_init_interval_hours": 12,
                    "truth_interval_hours": 6,
                    "truth_point_count": 21,
                    "init_grade": GRADE_LABELS[init_point.grade],
                    "init_lat_deg_n": init_point.lat,
                    "init_lon_deg_e": init_point.lon,
                    "init_central_pressure_hpa": init_point.central_pressure_hpa,
                    "init_max_wind_kt": init_point.max_wind_kt,
                    "end_grade": GRADE_LABELS[end_point.grade],
                    "end_lat_deg_n": end_point.lat,
                    "end_lon_deg_e": end_point.lon,
                    "end_central_pressure_hpa": end_point.central_pressure_hpa,
                    "end_max_wind_kt": end_point.max_wind_kt,
                    "quality_status": "PASS_21_CONTINUOUS_TROPICAL_POINTS",
                }
            )

            for lead in range(0, 121, 6):
                valid = init + timedelta(hours=lead)
                point = tropical_by_time[valid]
                truth_rows.append(
                    {
                        "case_id": case_id,
                        "TC_ID": tc_id,
                        "name": first_ts_point.name,
                        "case_seq_within_storm": case_seq,
                        "init_utc": iso_hour(init),
                        "lead_hours": lead,
                        "valid_utc": iso_hour(valid),
                        "jma_grade_code": point.grade,
                        "jma_grade": GRADE_LABELS[point.grade],
                        "jma_lat_deg_n": point.lat,
                        "jma_lon_deg_e": point.lon,
                        "jma_central_pressure_hpa": point.central_pressure_hpa,
                        "jma_max_wind_kt": point.max_wind_kt,
                        "truth_status": "VALID_TROPICAL_POINT",
                    }
                )

        for point in points:
            track_rows.append(
                {
                    "TC_ID": tc_id,
                    "name": point.name,
                    "time_utc": iso_hour(point.time),
                    "jma_grade_code": point.grade,
                    "jma_grade": GRADE_LABELS.get(point.grade, f"GRADE_{point.grade}"),
                    "lat_deg_n": point.lat,
                    "lon_deg_e": point.lon,
                    "central_pressure_hpa": point.central_pressure_hpa,
                    "max_wind_kt": point.max_wind_kt,
                    "is_ts_or_higher": point.grade in TS_OR_HIGHER,
                    "is_tropical": point.grade in TROPICAL_GRADES,
                    "source_line": point.source_line,
                }
            )

    if len(storm_rows) != 27:
        raise AssertionError(f"Expected 27 storm rows, got {len(storm_rows)}")
    if len(case_rows) != 74:
        raise AssertionError(f"Expected 74 cases, got {len(case_rows)}")
    if len(truth_rows) != 74 * 21:
        raise AssertionError(f"Expected 1554 truth rows, got {len(truth_rows)}")
    if any(row["truth_point_count"] != 21 for row in case_rows):
        raise AssertionError("All cases must have exactly 21 truth points")

    write_csv(
        args.output_dir / "storm_summary.csv",
        storm_rows,
        list(storm_rows[0]),
    )
    write_csv(args.output_dir / "case_catalog.csv", case_rows, list(case_rows[0]))
    write_csv(
        args.output_dir / "case_truth_6h.csv", truth_rows, list(truth_rows[0])
    )
    write_csv(
        args.output_dir / "jma_2025_track_points.csv", track_rows, list(track_rows[0])
    )

    metadata = {
        "generated_at_utc": iso_hour(datetime.now(UTC)),
        "source_file": str(args.jma_zip.resolve()),
        "source_url": SOURCE_URL,
        "source_sha256": file_sha256(args.jma_zip),
        "source_member": headers["_archive"],
        "protocol": {
            "storm_scope": "JMA international IDs 2501--2527",
            "first_ts_definition": "first JMA grade in {3,4,5}",
            "first_init_definition": "first 06/18 UTC cycle not earlier than first TS",
            "rolling_init_interval_hours": 12,
            "forecast_hours": 120,
            "truth_interval_hours": 6,
            "required_truth_points_per_case": 21,
            "accepted_tropical_grades": [2, 3, 4, 5],
            "excluded_grades": [6, 7, 9],
        },
        "counts": {
            "storms": len(storm_rows),
            "storms_with_multiple_cases": sum(
                row["complete_120h_case_count"] > 1 for row in storm_rows
            ),
            "storms_with_one_case": sum(
                row["complete_120h_case_count"] == 1 for row in storm_rows
            ),
            "storms_with_zero_cases": sum(
                row["complete_120h_case_count"] == 0 for row in storm_rows
            ),
            "cases": len(case_rows),
            "truth_rows": len(truth_rows),
            "track_rows": len(track_rows),
        },
        "outputs": {
            "storm_summary": "storm_summary.csv",
            "case_catalog": "case_catalog.csv",
            "case_truth_6h": "case_truth_6h.csv",
            "jma_track_points": "jma_2025_track_points.csv",
        },
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
