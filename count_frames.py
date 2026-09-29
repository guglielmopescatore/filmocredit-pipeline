import csv
from collections import defaultdict
from pathlib import Path

base = Path("data/episodes")
output = Path("frame_counts.csv")

SUFFIXES = ("_Opening", "_End")

counts = defaultdict(int)
for episode_dir in sorted(base.iterdir()):
    if not episode_dir.is_dir():
        continue
    name = episode_dir.name
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    frames_dir = episode_dir / "analysis" / "frames"
    if frames_dir.exists():
        counts[name] += sum(1 for _ in frames_dir.glob("*.jpg"))

rows = sorted(counts.items())

with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "jpg_count"])
    writer.writerows(rows)

print(f"Written to {output} ({len(rows)} episodes)")
for name, count in rows:
    print(f"  {name}: {count}")
