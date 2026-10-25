"""Audit actual JMA tracks and the grade 2–5 subset used for turning detection."""
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / 'inputs' / 'jma_track_points.csv'
META = HERE / 'inputs' / 'jma_source_metadata.json'
rows = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
rows = [r for r in rows if 2501 <= int(r['TC_ID']) <= 2527]
assert {int(r['TC_ID']) for r in rows} == set(range(2501, 2528))
metadata = json.loads(META.read_text())
archive = HERE / 'inputs' / 'jma_original_bst_all_20260723.zip'
raw_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
assert raw_hash == metadata['source_sha256']
with zipfile.ZipFile(archive) as z:
    raw_lines = {line.rstrip() for line in z.read(metadata['source_member']['member_name']).decode().splitlines()}
assert all(r['source_line'].rstrip() in raw_lines for r in rows)

def summarize(records):
    groups = defaultdict(list)
    crossings = []
    for r in records:
        groups[r['TC_ID']].append(r)
    for tc, group in sorted(groups.items()):
        group.sort(key=lambda r: r['time_utc'])
        for a, b in zip(group, group[1:]):
            x, y = float(a['lon_deg_e']), float(b['lon_deg_e'])
            # Signed longitude jumps identify the antimeridian, including
            # source longitudes expressed continuously above 180 degrees.
            nx, ny = (x + 180) % 360 - 180, (y + 180) % 360 - 180
            if abs(ny - nx) > 180:
                crossings.append(dict(TC_ID=tc, name=b['name'],
                    from_utc=a['time_utc'], to_utc=b['time_utc'],
                    from_lon=x, to_lon=y,
                    from_grade=int(a['jma_grade_code']), to_grade=int(b['jma_grade_code'])))
    return dict(n_cyclones=len(groups), n_records=len(records),
                longitude_min=min(float(r['lon_deg_e']) for r in records),
                longitude_max=max(float(r['lon_deg_e']) for r in records),
                crossings=crossings)

report = dict(source=str(SOURCE.relative_to(HERE)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              raw_archive=str(archive.relative_to(HERE)), raw_archive_sha256=raw_hash,
              all_882_source_lines_verified=True,
              complete_jma_tracks=summarize(rows),
              turning_input_grades_2_to_5=summarize([r for r in rows if int(r['jma_grade_code']) in {2,3,4,5}]))
out = HERE / 'data/jma_dateline_audit.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
