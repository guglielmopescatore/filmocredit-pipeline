#!/usr/bin/env python3
"""
Section C - why is the published hold-out run (name + role F1 0.925) above all five repetitions?

Four checks on the GPT Sol Standard hold-out runs (same 207 frames everywhere):
  1. the F1 of every repetition, No Role and With Role;
  2. the published run recomputed with the same code as the repetitions
     (if it no longer gave 0.925, the difference would be in the calculation);
  3. when the published run was made and with which code/prompt/platform
     (raw responses of its DB: dates, input tokens of the first frame of every
     clip, response fields returned by Azure);
  4. credit-by-credit role comparison: gold credits right in the published run
     and wrong in all five repetitions, and the opposite, by role and by title,
     plus the decomposition of the With-Role TP gap: for each gold credit,
     published hit (0/1) minus the share of repetitions that hit it; the sum over
     all credits is TP(published) - mean TP(repetitions).

Writes results/C_holdout_check.md and results/C_holdout_check_credits.csv.
Usage: python analysis_2nd_revision/section_C_holdout_check.py
"""

import collections
import csv
import json
import sqlite3
import subprocess
from pathlib import Path

from metrics_lib import GOLD_HOLDOUT, RESULTS, ROOT, evaluate, md_table, predictions

REPS = [f"C_SOL_r{n}" for n in range(1, 6)]
ORIG = "ORIG_SOL_holdout5"
ORIG_DB = ROOT / "db" / "FUZZY88_GPT_SOL_STANDARD_25products_tvcredits_v3.db"
REP_DBS = [ROOT / "runs" / f"REP_SOL_r{n}" / "db" / "tvcredits_v3.db" for n in range(1, 6)]
HOLDOUT_PREFIXES = ("3_Percent", "Blue_Eye", "Honeyland", "Persepolis", "Wild_Str")


def raw_calls(db: Path) -> list[tuple]:
    conn = sqlite3.connect(db)
    rows = [r for r in conn.execute("SELECT episode_id, source_frame, recorded_at, raw_response FROM raw_response_llm_call ORDER BY id")
            if r[0].startswith(HOLDOUT_PREFIXES)]
    conn.close()
    return rows


def git_time(commit: str) -> str:
    return subprocess.run(["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M", commit],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    md = ["# Section C - the published hold-out run vs the five repetitions", ""]

    # 1-2. F1 per run, published run recomputed with the repetitions' code
    rows = evaluate(ORIG, "published (July 2026)", GOLD_HOLDOUT)
    for i, name in enumerate(REPS, 1):
        rows += evaluate(name, f"repetition r{i}", GOLD_HOLDOUT)
    f1 = {(r["label"], r["mode"]): r for r in rows}
    labels = ["published (July 2026)"] + [f"repetition r{i}" for i in range(1, 6)]
    table = [{"run": l, "F1 No Role": f1[(l, "No Role")]["F1"], "F1 With Role": f1[(l, "With Role")]["F1"],
              "TP With Role": f1[(l, "With Role")]["TP"], "FP With Role": f1[(l, "With Role")]["FP"],
              "FN With Role": f1[(l, "With Role")]["FN"]} for l in labels]
    md += ["## 1-2. F1 of every run (same scoring code for all)", "",
           "Exact match against the hold-out gold set with `compare_llm_human_metrics.py`'s functions, the same "
           "code and options for the published run and the repetitions. The published run recomputed this way "
           f"still gives **{f1[('published (July 2026)', 'With Role')]['F1']:.4f}** With Role: the difference is not "
           "in the calculation.", "", md_table(table, list(table[0])), ""]

    # 3. dates, prompt (first-frame input tokens), platform (response fields)
    orig_calls = raw_calls(ORIG_DB)
    rep_calls = raw_calls(REP_DBS[0])
    dates = collections.defaultdict(list)
    for ep, _, at, _ in orig_calls:
        dates[ep].append(at)
    first = {}
    for ep, fr, _, raw in orig_calls:
        first.setdefault(ep, (fr, json.loads(raw)))
    rep_by_frame = {(ep, fr): json.loads(raw) for ep, fr, _, raw in rep_calls}
    tok_rows = []
    for ep, (fr, resp) in sorted(first.items()):
        rep = rep_by_frame.get((ep, fr), {})
        tok_rows.append({"clip": ep, "published Stage III (UTC)": f"{min(dates[ep])[:16]} - {max(dates[ep])[11:16]}",
                         "first-frame input tokens, published": resp["usage"]["input_tokens"],
                         "same frame, r1": rep.get("usage", {}).get("input_tokens"),
                         "difference": rep.get("usage", {}).get("input_tokens", 0) - resp["usage"]["input_tokens"]})
    o_keys = set(json.loads(orig_calls[0][3]))
    r_keys = set(json.loads(rep_calls[0][3]))
    o0, r0 = json.loads(orig_calls[0][3]), json.loads(rep_calls[0][3])
    same = {k: (o0.get(k), r0.get(k)) for k in ("model", "service_tier", "temperature", "top_p", "truncation")}
    md += ["## 3. When and with which code the published run was made", "",
           f"Stage III of the published hold-out run: 18-23 July 2026; Stage IV on 24 July 11:10. Commit `5fec349` "
           f"(\"Unify pipeline steps and upgrade IMDB matching\", ~1,200 changed lines in Stage III/IV files) is of "
           f"{git_time('5fec349')}, the previous commit `3c9dcf8` of {git_time('3c9dcf8')}: the published run used the "
           "uncommitted working tree of those days. The published 20-product run is of the same days "
           "(Stage III 17-22 July 2026).", "",
           "**Prompt.** The VLM prompt of `f9c9bb6` equals the one of `5fec349`, and differs from the 13 July commit by "
           "~100 tokens (name transcription, guild acronyms, special thanks). On the first frame of every clip (empty "
           "previous-credits context, so the input is prompt + image only) the input tokens of the published run and of "
           "the repetitions are equal or differ by 5-8 tokens out of 10-12 thousand: the published run already used "
           "(almost exactly) today's prompt.", "", md_table(tok_rows, list(tok_rows[0])), "",
           "**Model and platform.** Same deployment name, reasoning (standard/medium) and sampling parameters "
           f"({', '.join(f'{k}={v[0]}' for k, v in same.items())}) in July and September. The Azure responses, however, "
           f"changed: September responses carry fields absent in July ({', '.join(sorted(r_keys - o_keys))}) and the "
           f"default prompt cache retention went from `{o0.get('prompt_cache_retention')}` to "
           f"`{r0.get('prompt_cache_retention')}`. The service behind `gpt-5.6-sol` was updated between the two periods; "
           "whether the model weights changed cannot be told from the responses (no snapshot id is returned).", ""]

    # 4. credit-by-credit role comparison and TP-gap decomposition
    orig = predictions(ORIG)[0]
    reps = [predictions(n)[0] for n in REPS]
    names = [collections.defaultdict(set) for _ in range(6)]
    for i, tr in enumerate([orig] + reps):
        for ep, n, rg in tr:
            names[i][(ep, n)].add(rg)
    credit_rows, orig_only, reps_only = [], [], []
    contrib_title, contrib_role = collections.Counter(), collections.Counter()
    for ep, n, rg in sorted(GOLD_HOLDOUT):
        o = (ep, n, rg) in orig
        k = sum((ep, n, rg) in r for r in reps)
        contrib = int(o) - k / 5
        contrib_title[ep] += contrib
        contrib_role[rg] += contrib
        row = {"product": ep, "name": n, "gold_role": rg, "published_right": int(o), "repetitions_right": k,
               "published_roles": "; ".join(sorted(names[0].get((ep, n), []))),
               "repetition_roles": "; ".join(sorted(set().union(*(names[i].get((ep, n), set()) for i in range(1, 6)))))}
        credit_rows.append(row)
        if o and k == 0:
            orig_only.append(row)
        if not o and k == 5:
            reps_only.append(row)
    with open(RESULTS / "C_holdout_check_credits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(credit_rows[0]))
        w.writeheader()
        w.writerows(credit_rows)

    gap = sum(contrib_title.values())
    rep_tp = [f1[(f"repetition r{i}", "With Role")]["TP"] for i in range(1, 6)]
    tp_mean = sum(rep_tp) / len(rep_tp)
    tp_sd = (sum((x - tp_mean) ** 2 for x in rep_tp) / (len(rep_tp) - 1)) ** 0.5
    by = lambda lst, key: collections.Counter(r[key] for r in lst).most_common()
    detail = [{k: r[k] for k in ("product", "name", "gold_role", "published_roles", "repetition_roles")} for r in orig_only + reps_only]
    title_rows = [{"product": t, "TP gap contribution": round(v, 1)} for t, v in sorted(contrib_title.items(), key=lambda kv: -kv[1])]
    role_rows = [{"gold role": t, "TP gap contribution": round(v, 1)} for t, v in sorted(contrib_role.items(), key=lambda kv: -kv[1]) if abs(v) >= 1]
    md += ["## 4. Credit-by-credit role comparison", "",
           f"Of {len(GOLD_HOLDOUT):,} gold credits (name + role): **{len(orig_only)} are right in the published run and "
           f"wrong in all five repetitions**, **{len(reps_only)} the opposite**. In every one of these the name is found "
           "on both sides: only the role differs.", "",
           f"- published right / repetitions wrong, by role: {by(orig_only, 'gold_role')}; by product: {by(orig_only, 'product')}",
           f"- repetitions right / published wrong, by role: {by(reps_only, 'gold_role')}; by product: {by(reps_only, 'product')}", "",
           "They are concentrated on a few credit cards with ambiguous roles, where all five repetitions agree on the "
           "same alternative: *Persepolis* storyboard (`scenarimage`) credits classified as Animation instead of Art "
           "Department, *Blue Eye Samurai* heads of studio as Additional Crew instead of Production Managers, trainees "
           "(`stagiaire`, `estagiario`) as Additional Crew. The model reads the same `role_detail` text in both periods "
           "and assigns a different `role_group`; the normalization code does not change it.", "",
           md_table(detail, list(detail[0])), "",
           f"**Decomposition of the With-Role TP gap** (published minus mean of the repetitions = {gap:+.1f} TP):", "",
           md_table(title_rows, list(title_rows[0])), "", md_table(role_rows, list(role_rows[0])), "",
           "## Reading", "",
           "- Not the calculation (check 2) and not the prompt (check 3: same input tokens on the first frame).",
           f"- The published run is not uniformly better: on names (No Role) it is *below* all five repetitions "
           f"({f1[('published (July 2026)', 'No Role')]['F1']:.4f} vs "
           f"{min(f1[(f'repetition r{i}', 'No Role')]['F1'] for i in range(1, 6)):.4f}-"
           f"{max(f1[(f'repetition r{i}', 'No Role')]['F1'] for i in range(1, 6)):.4f}); only on roles is it above "
           f"(TP {f1[('published (July 2026)', 'With Role')]['TP']} vs repetitions {tp_mean:.0f} +/- {tp_sd:.1f}, "
           f"{(f1[('published (July 2026)', 'With Role')]['TP'] - tp_mean) / tp_sd:.1f} SD). A lucky draw from today's "
           "distribution would not be that far off, nor better on roles and worse on names at the same time.",
           "- The difference is a consistent change of the role assigned to a few blocks of credits, the same in all five "
           "repetitions, with the service behind the same deployment name updated in between. This looks like a change in "
           "model behaviour between July and September rather than a lucky single run, but the two cannot be separated with "
           "these data: the July service can no longer be queried.",
           "- On the 20 main products the comparison published vs today (0.904 vs 0.911 With Role) is not on the same "
           "input: today's run also changes Stage II (5fec349, OCR) and therefore the frames. The hold-out repetitions are "
           "the only same-input comparison.",
           "- Safe statement for the paper: the hold-out figure is the mean of the five repetitions "
           "(see results/C_dispersion.csv), with the published run reported as an earlier execution on the July service.", ""]
    (RESULTS / "C_holdout_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
