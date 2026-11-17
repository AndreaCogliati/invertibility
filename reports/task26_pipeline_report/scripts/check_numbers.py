"""TASK-26 report: check the report's key numbers against saved artifacts (read-only).

Each entry: (claim as printed in main.tex, value read from the source, printed precision). The script
also asserts that the printed string occurs in main.tex. Exit code 1 on any mismatch.

    uv run python reports/task26_pipeline_report/scripts/check_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TEX = (Path(__file__).resolve().parents[1] / "main.tex").read_text()
A = REPO / "artifacts"


def j(p: str) -> dict:
    return json.loads((A / p).read_text())


geo = json.loads((Path(__file__).parent / "geometry.json").read_text())
fig = json.loads((Path(__file__).parent / "figure_data.json").read_text())
t15n2, t15n8 = j("task15_bigvgan_memorization/n2/report.json"), j("task15_bigvgan_memorization/n8/report.json")
t16 = j("task16_bigvgan_corrected/n2/report.json")
t18 = j("task18_heldout/task18_heldout_v1/report.json")
t19 = j("task19/analysis_v1/analysis.json")["table"]
t20 = j("task20/analysis_v1/analysis.json")["table"]
t21 = j("task21/analysis_v1/analysis.json")["table"]
t22 = next(c for c in j("task22/analysis_v1/analysis.json")["comparisons"] if c["name"].startswith("window best"))["per_set"]
t24 = j("task24/task24_v1/report.json")
t25 = fig["task25"]

checks = [
    ("47{,}872", geo["crop"]["num_samples"], 0),
    ("(1, 128, 199)", tuple(geo["muq"]["frontend_out_shape"]), None),
    ("(1, 50, 1024)", tuple(geo["muq"]["hidden_state_shapes"]["1"]), None),
    ("(1, 100, 187)", tuple(geo["target_mel"]["shape"]), None),
    ("1{,}271{,}908", geo["translator"]["current"]["params"], 0),
    ("4{,}509{,}796", geo["translator"]["wide_x2"]["params"], 0),
    ("112{,}414{,}512", geo["bigvgan"]["params"], 0),
    ("333{,}401{,}472", geo["muq"]["params"], 0),
    ("6.7684", geo["muq"]["frontend"]["stat_values"]["melspec_2048_mean"], 4),
    ("18.4179", geo["muq"]["frontend"]["stat_values"]["melspec_2048_std"], 4),
    ("$-11.513$", geo["bigvgan"]["log_mel_floor"], 3),
    ("805/18{,}700", geo["target_mel"]["bins_at_floor"], 0),
    ("std 1.72", geo["muq"]["block1_std"], 2),
    ("0.90", geo["muq"]["last_std"], 2),
    ("138.7", geo["translator"]["receptive_field_seconds"] * 1000, 1),
    ("0.133", geo["alignment"]["positions_first3_last3"][0], 3),
    ("0.024 at step 4{,}750", t15n2["best"]["summary"]["mel_l1_pred_vs_target"], 3),
    ("0.032 (correlation 0.92", t15n8["best"]["summary"]["mel_l1_pred_vs_target"], 3),
    ("shuffled-across control 1.20", t15n2["best"]["summary"]["mel_l1_shuffled_vs_target"], 2),
    ("round trip is 0.109", t15n8["best"]["summary"]["oracle_resynthesis_mel_l1"], 3),
    ("memorized 0.354", t18["best"]["memorized"]["conditions"]["correct"]["mel_l1"]["mean"], 3),
    ("val 0.576", t18["best"]["val"]["conditions"]["correct"]["mel_l1"]["mean"], 3),
    ("test 0.525", t18["best"]["test"]["conditions"]["correct"]["mel_l1"]["mean"], 3),
    ("0.550 $\\to$ 0.501", t19["primary_heldout"]["A32"]["best"], 3),
    ("0.550 $\\to$ 0.501", t19["primary_heldout"]["A128"]["best"], 3),
    ("val 0.475, test 0.465, val2 0.481, test2 0.494", t20["val (primary)"]["N512_best"], 3),
    ("val 0.475, test 0.465, val2 0.481, test2 0.494", t20["test2 (secondary)"]["N512_best"], 3),
    ("memorized 0.396 $\\to$ 0.441", t20["memorized (core 32)"]["A128_best"], 3),
    ("memorized 0.396 $\\to$ 0.441", t20["memorized (core 32)"]["N512_best"], 3),
    ("val 0.455 (inflated", t21["val"]["T21best"], 3),
    ("test 0.454", t21["test"]["T21best"], 3),
    ("val2 0.473", t21["val2"]["T21best"], 3),
    ("test2 0.480", t21["test2"]["T21best"], 3),
    ("0.368 final", t21["memorized (core 32)"]["T21final"], 3),
    ("0.391 at best", t21["memorized (core 32)"]["T21best"], 3),
    ("& 0.455 & 0.260 &", t22["val (primary validation, unseen tracks)"]["mean_b"], 3),
    ("& 0.480 & 0.263 &", t22["test2 (secondary test, unseen tracks)"]["mean_b"], 3),
    ("& 0.454 & 0.257 &", t22["test (primary test, unseen tracks)"]["mean_b"], 3),
    ("0.0002/0.041/0.095/0.174/0.243", fig["task24"]["256"]["fit_final"], 3),
    ("Monitoring 1.34/1.13/0.66/0.44/0.31", fig["task24"]["1"]["monitor_final"], 2),
    ("Monitoring 1.34/1.13/0.66/0.44/0.31", fig["task24"]["256"]["monitor_final"], 2),
    ("Fit: A 0.230, B 0.167, C 0.170, D 0.206, E 0.145, F 0.203", t25["A"]["fit_final"], 3),
    ("Fit: A 0.230, B 0.167, C 0.170, D 0.206, E 0.145, F 0.203", t25["E"]["fit_final"], 3),
    ("Fit: A 0.230, B 0.167, C 0.170, D 0.206, E 0.145, F 0.203", t25["F"]["fit_final"], 3),
    ("Monitoring: A 0.308, D 0.311, F 0.315, B 0.361, E 0.364, C 0.375", t25["C"]["monitor_final"], 3),
    ("Monitoring: A 0.308, D 0.311, F 0.315, B 0.361, E 0.364, C 0.375", t25["A"]["monitor_final"], 3),
    ("0.37157 vs 0.37158", fig["anchor"]["task21_best_l1_regenerated"], 5),
    ("TASK-22 0.149", fig["anchor"]["temporal_variation"]["TASK-22 best (block 1, step 193k)"], 3),
    ("$V=0.254$", fig["anchor"]["temporal_variation"]["target"], 3),
    ("D 0.57, F 0.58, A 0.63", fig["task25_temporal_variation_ratio"]["D"]["monitor16"], 2),
    ("E 0.69, B 0.75, C 0.83", fig["task25_temporal_variation_ratio"]["C"]["monitor16"], 2),
]
anchor_ladder = t24["summaries"] if "summaries" in t24 else None

bad = 0
for text, value, prec in checks:
    printed = text in TEX
    if prec is None:
        shown = str(value).replace(" ", "")
        ok = printed and shown.strip("()") in text.replace(" ", "").strip("()")
    else:
        shown = f"{value:,.{prec}f}".replace(",", "{,}") if prec == 0 else f"{value:.{prec}f}"
        ok = printed and shown.lstrip("-") in text.replace("$", "")
    bad += not ok
    print(("OK  " if ok else "FAIL") + f"  {text!r:70} source={shown}")
print(f"{len(checks) - bad}/{len(checks)} checks passed")
sys.exit(1 if bad else 0)
