"""Stage II determinism check.

Part 1 (no rerun): static scenes do not depend on the scroll cap, so for them
PIPE_cap150 / cap100 / cap75 are three independent Stage II repetitions.
Part 2 (rerun): rerun Stage II with the PIPE_cap150 code on a few episodes in a
scratch folder and compare with the PIPE_cap150 output.
"""
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

MAIN = Path(r"D:\DATA\dev\filmocredit")
RUN = MAIN / "runs" / "PIPE_cap150"
OUT = MAIN / "runs" / "_stage2_rerun"  # scratch, git-ignored
RERUN_EPISODES = sys.argv[1:] or ["El_desorden_que_dejas_S01E03_End", "Planet_Earth_S01E10", "La_piovra_1x2"]
NUM = re.compile(r"_num(\d+)")


def scene_frames(analysis_dir: Path) -> dict[str, tuple[str, list[int]]]:
    manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))["scenes"]
    on_disk = set(os.listdir(analysis_dir / "frames"))
    return {
        k: (s.get("type"), sorted(int(NUM.search(f["path"]).group(1)) for f in s.get("output_files", [])
                                  if f.get("path") and Path(f["path"]).name in on_disk))
        for k, s in manifest.items()
    }


def part1() -> None:
    episodes = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))["reference_frames"]
    agg = Counter()
    for ep in episodes:
        runs = {c: scene_frames(MAIN / "runs" / f"PIPE_cap{c}" / "data" / "episodes" / ep / "analysis") for c in (150, 100, 75)}
        for k, (typ, frames) in runs[150].items():
            if typ == "dynamic_scroll":
                continue
            same = all(runs[c].get(k) == (typ, frames) for c in (100, 75))
            agg["identical" if same else "DIFFERENT"] += 1
            if not same:
                print("  static scene differs across caps:", ep, k, [runs[c].get(k) for c in (150, 100, 75)])
    print("Part 1 - non-scroll scenes across cap150/100/75:", dict(agg))


def part2() -> None:
    os.chdir(RUN)
    sys.path.insert(0, str(RUN))
    sys.path.insert(0, str(MAIN))
    import logging
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    from scripts_v3 import config, utils
    import run_headless

    assert config.SCROLL_MAX_FRAMES_PER_SAVE == 150
    shutil.rmtree(OUT, ignore_errors=True)
    config.EPISODES_BASE_DIR = OUT
    lang = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))["parameters"]["ocr_language"]
    print(f"Part 2 - OCR language of {RUN.name}: {lang}")
    reader = utils.get_paddleocr_reader(lang=config.PADDLEOCR_LANG_MAP.get(lang, lang))
    stopwords = utils.load_user_stopwords()
    for ep in RERUN_EPISODES:
        dst = OUT / ep / "analysis"
        dst.mkdir(parents=True)
        shutil.copy2(RUN / "data" / "episodes" / ep / "analysis" / "initial_scene_analysis.json", dst)
        run_headless.stage2_episode(ep, reader, stopwords)
        a = scene_frames(RUN / "data" / "episodes" / ep / "analysis")
        b = scene_frames(dst)
        diff = {k: (a[k], b.get(k)) for k in a if a[k] != b.get(k)}
        n_a = sum(len(v[1]) for v in a.values())
        n_b = sum(len(v[1]) for v in b.values())
        print(f"Part 2 - {ep}: PIPE_cap150 {n_a} frames, rerun {n_b} frames, "
              f"scenes {len(a)}, differing scenes {len(diff)}")
        for k, (x, y) in diff.items():
            print("    ", k, "run:", x, "rerun:", y)


if __name__ == "__main__":
    part1()
    part2()
