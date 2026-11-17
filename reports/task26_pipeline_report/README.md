# TASK-26 synthesis report: MuQ → mel → BigVGAN, through TASK-25

Brief: `tasks/TASK-26.md`. Documentation only: no training, and no existing result was modified.

## Contents

| Path | What it is |
| --- | --- |
| `main.tex`, `main.pdf` | Report source and compiled PDF (17 pages: about 15 of main text plus appendix) |
| `refs.bib` | Bibliography. MuQ and BigVGAN arXiv identifiers were checked on 2026-11-06; GTZAN; Sturm 2013 fault list |
| `figures/pipeline.tex` | Fig. 1, full pipeline (TikZ) |
| `figures/timegrid.tex` | Fig. 2, temporal grids (TikZ, drawn to scale) |
| `figures/timeline.tex` | Fig. 3, experiment sequence (TikZ) |
| `figures/*.pdf` | Figs. 4–8, data figures from saved artifacts (matplotlib, vector) |
| `scripts/inspect_geometry.py` → `scripts/geometry.json` | Measured shapes, rates, centers and parameter counts on the TASK-23 anchor crop |
| `scripts/make_figures.py` → `scripts/figure_data.json` | Data figures, plus the numbers they show |
| `scripts/check_numbers.py` | Asserts 51 printed numbers against saved artifacts (51/51 pass) |

## Build (from the repository root)

```bash
uv run python reports/task26_pipeline_report/scripts/inspect_geometry.py --device cpu   # needs local MuQ weights + third_party/ BigVGAN
uv run python reports/task26_pipeline_report/scripts/make_figures.py                    # needs local TASK-21/22 checkpoints, TASK-23..25 preds/logs
uv run python reports/task26_pipeline_report/scripts/check_numbers.py
cd reports/task26_pipeline_report && latexmk -pdf main.tex
```

Set `HF_HUB_OFFLINE=1` if the MuQ weights are already cached. The scripts write only inside this directory.

The PDF compiles with TeX Live 2026 (`pdflatex` via `latexmk`) with no undefined references. Four overfull boxes of at most 3 pt remain, all from long monospace identifiers.

## Data provenance for the figures

- **Saved predictions:**
  - TASK-23 B best (`runs/task23/task23_v1/B/preds/best.pt`);
  - TASK-24 N=256 final;
  - TASK-25 A–F final (showcase row 0 = anchor; 16 monitoring crops).
- **Regenerated predictions** (deterministic CPU forward pass of the stored best checkpoints): TASK-21 best (`runs/task21/task21_N512ext_v1/checkpoints/best.pt`, step 60,000) and TASK-22 best (step 193,000) on the anchor crop.
  - Their mel L1 matches the recorded per-example values: 0.371573 vs 0.371579, and 0.2383648 vs 0.2383647.
  - TASK-21's input (`last_hidden_state`) was re-extracted on CPU, because its ACES feature cache is not local.
- **New descriptive statistic:** the temporal-variation statistic V (mean frame-to-frame change of a mel) was computed for this report. It is not a training metric.

## Unresolved gaps

- No commit identifier is recorded in the local artifacts for TASK-15, TASK-16 or the TASK-18 full run.
- Listening:
  - pending for TASK-16, TASK-22 (the blinded package is unrated) and TASK-24;
  - none for TASK-21;
  - informal and unblinded for TASK-23 and TASK-25.
- Seed variability is unmeasured for every contrast.
- The source of the click in BigVGAN(ground-truth mel) on the anchor is unresolved.
- Not measured:
  - how well MuQ tokens are localized in time after self-attention;
  - the effect of isolated-crop extraction on edge tokens;
  - any alignment-offset sweep.
- The anchor track's source file has a full-scale glitch at 7.18 s. Resampled, it peaks at 1.99 and sets the whole-track normalization gain (×0.477). Whether such outliers matter elsewhere has not been surveyed.
- No W&B run was created: this task runs no experiment. The scripts are read-only inspections of saved artifacts.
