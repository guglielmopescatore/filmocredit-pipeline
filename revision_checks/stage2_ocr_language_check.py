"""Does the reference Stage II come from a different OCR language?

PaddleOCR's English recogniser (en_PP-OCRv5_mobile_rec) was downloaded on
2025-12-04, the day of the reference Stage II, while the runs of today use
lang="it" (latin_PP-OCRv5_mobile_rec). Rerun Stage II with the PIPE_cap150 code
on a few episodes with each language and compare the selected frame numbers
with the reference (data/episodes/<ep>/analysis of the main checkout).

Usage:
    python revision_checks/stage2_ocr_language_check.py [--run PIPE_cap150] [--langs en it] [episode ...]
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent
OUT = MAIN / "runs" / "_stage2_lang_check"  # scratch, git-ignored
NUM = re.compile(r"_num(\d+)")


def frame_numbers(analysis_dir: Path) -> set[int]:
    manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))["scenes"]
    on_disk = set(os.listdir(analysis_dir / "frames"))
    return {
        int(NUM.search(f["path"]).group(1))
        for s in manifest.values() for f in s.get("output_files", [])
        if f.get("path") and Path(f["path"]).name in on_disk
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="PIPE_cap150", help="worktree whose code, clips and Stage I are used")
    parser.add_argument("--langs", nargs="+", default=["en", "it"])
    parser.add_argument("episodes", nargs="*", default=["El_desorden_que_dejas_S01E03_End", "Psycho"])
    args = parser.parse_args()
    RUN = MAIN / "runs" / args.run
    os.chdir(RUN)
    sys.path.insert(0, str(RUN))
    sys.path.insert(0, str(MAIN))
    import logging
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    from scripts_v3 import config, utils
    import run_headless

    assert config.SCROLL_MAX_FRAMES_PER_SAVE == 150
    stopwords = utils.load_user_stopwords()
    for lang in args.langs:
        reader = utils.get_paddleocr_reader(lang=config.PADDLEOCR_LANG_MAP.get(lang, lang))
        config.EPISODES_BASE_DIR = OUT / lang
        for ep in args.episodes:
            dst = OUT / lang / ep / "analysis"
            shutil.rmtree(dst.parent, ignore_errors=True)
            dst.mkdir(parents=True)
            shutil.copy2(RUN / "data" / "episodes" / ep / "analysis" / "initial_scene_analysis.json", dst)
            run_headless.stage2_episode(ep, reader, stopwords)
            now = frame_numbers(dst)
            ref = frame_numbers(MAIN / "data" / "episodes" / ep / "analysis")
            print(f"lang={lang} {ep}: rerun {len(now)} frames, reference {len(ref)}, "
                  f"common {len(now & ref)}, only_rerun {len(now - ref)}, only_reference {len(ref - now)}")


if __name__ == "__main__":
    main()
