"""Renders TASK-18's report.json as Markdown (TASK-18_RESULTS.md). Pure function of the report."""

from __future__ import annotations

import math

from muq_inversion.task18 import CONDITIONS, regime_status

REGIME_TITLES = {
    "memorized": "1. memorized training crops (not evidence of invertibility)",
    "train_unseen": "2. unseen crops from known training tracks",
    "val": "3. validation tracks (unseen; used for checkpoint selection)",
    "test": "4. test tracks (unseen; never used for any decision)",
    "memorized_new": "memorized crops of the 96 TASK-19 training tracks (unseen tracks for 32-track arms)",
    "train_unseen_new": "unseen crops of the 96 TASK-19 training tracks (unseen tracks for 32-track arms)",
    "val2": "secondary validation tracks (24; not used for selection)",
    "test2": "secondary test tracks (24; never used for any decision)",
}
SECONDARY = ["mel_cosine", "mel_pearson", "mel_variance_ratio", "rms_corr", "centroid_corr", "chroma_cosine",
             "onset_corr_0ms", "onset_corr_50ms", "onset_corr_200ms", "onset_corr_500ms", "resynthesis_mel_l1",
             "stft_distance", "waveform_correlation_phase_dependent"]


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{nd}f}"


def render_results(report: dict) -> str:
    best, outcome = report["best"], report["outcome_best"]
    cfg = report["resolved_config"]
    sets = list(best)
    status = {s: outcome["regimes"].get(s) or regime_status(best[s], cfg["outcome_rule"]) for s in sets}
    lines = [
        f"# {report.get('task', 'TASK-18')} results: {report['run_name']}"
        + (f" (arm {report['arm']})" if report.get("arm") else ""),
        "",
        f"**Outcome (pre-registered rule, best checkpoint): `{outcome['code']}`.** {outcome['text']}",
        "",
        f"Listening verdict: **{outcome['listening_verdict']}**. {outcome['note']}",
        "",
        (f"- Best checkpoint: step {report['best_checkpoint']['step']} of {report['final_step']}, "
         f"selected by {report['best_checkpoint']['selection']}."),
        f"- Final-step outcome, for reference: `{report['outcome_final']['code']}`.",
        (f"- W&B run: `{report['wandb_run_id']}`. Device: {cfg['device_metadata']['device_name']} "
         f"({cfg['device_metadata']['backend']}, torch {cfg['device_metadata']['torch_version']}, FP32)."),
        f"- Manifests: tracks `{cfg['track_manifest_sha256'][:12]}`, crops `{cfg['crop_manifest_sha256'][:12]}`.",
        "",
        "## Primary endpoint: mel L1 (raw prediction vs official target), best checkpoint",
        "",
        "Lower is better. *Beats* = the track-clustered 95% CI of (control - correct) is entirely above 0.",
        "",
        "| Regime | crops / tracks | BigVGAN round trip | correct | " + " | ".join(c for c in CONDITIONS if c != "correct")
        + " | correct / round trip | beats all |",
        "| --- | --- | --- | --- | " + " | ".join("---" for c in CONDITIONS if c != "correct") + " | --- | --- |",
    ]
    for s in sets:
        a, st = best[s], status[s]
        cells = []
        for c in CONDITIONS:
            if c == "correct":
                continue
            d = a["paired_mel_l1"][c]
            mark = "✓" if st["beats"][c] else "✗"
            cells.append(f"{_f(a['conditions'][c]['mel_l1']['mean'])} {mark} (Δ {_f(d['mean_delta'])} "
                         f"[{_f(d['ci_low'])}, {_f(d['ci_high'])}])")
        lines.append(f"| {s} | {a['n_crops']} / {a['n_tracks']} | {_f(a['oracle_round_trip_mel_l1']['mean'])} | "
                     f"{_f(a['conditions']['correct']['mel_l1']['mean'])} | " + " | ".join(cells)
                     + f" | {_f(st['mel_l1_over_round_trip'], 2)} | {'yes' if st['beats_all'] else 'no'} |")
    lines += ["", "## Secondary metrics (mean; best checkpoint)", "",
              ("Correlations are Pearson r against the original audio (contours) or the target mel. "
               "`waveform_correlation_phase_dependent` needs sample-level phase agreement and is a diagnostic only."), ""]
    for s in sets:
        a = best[s]
        lines += [f"### {REGIME_TITLES.get(s, s)}", "", "| metric | oracle | " + " | ".join(CONDITIONS) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in CONDITIONS) + " |"]
        for k in SECONDARY:
            if k not in a["conditions"]["correct"]:
                continue
            oracle = a.get("oracle_contours", {}).get(k, {}).get("mean", float("nan"))
            lines.append(f"| {k} | {_f(oracle)} | " + " | ".join(_f(a["conditions"][c][k]["mean"]) for c in CONDITIONS) + " |")
        lines.append("")
    exp = report["exposure"]
    lines += [
        "## Training exposure", "",
        (f"{exp['tracks_visited']}/{exp['tracks_total']} training tracks visited; per-track draws "
         f"{exp['track_exposure_min']}-{exp['track_exposure_max']} (max/min {_f(exp['track_exposure_max_over_min'], 2)}); "
         f"{exp['pool_crops_visited']}/{exp['pool_crops_total']} pool crops visited."), "",
        "## Listening package (best checkpoint)", "",
        ("Each example has `original`, `bigvgan_oracle`, `correct`, `shuffled_across`, `shuffled_time`, `constant_zero` "
         "WAVs and a mel figure. Listen in that order; compare `correct` with the oracle and with the controls."), "",
        "| example | regime | mel L1 correct | round trip | folder |", "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {e['label']} | {e['eval_set']} | {_f(e['mel_l1_correct'])} | {_f(e['oracle_round_trip_mel_l1'])} | `{e['dir']}` |"
              for e in report["listening"]]
    lines += ["", "## Provenance", "",
              f"- Lineage: {report['lineage']}",
              f"- Paths: {report['paths']}", ""]
    return "\n".join(lines)
