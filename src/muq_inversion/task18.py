"""TASK-18: 32/8/8 held-out generalization of the corrected TASK-15/16 MuQ -> BigVGAN-mel pipeline.

Pure, testable pieces only: track/crop selection, the training sampler, conditioning controls,
mel and contour metrics, track-clustered statistics, and the pre-registered outcome rule. The
scientific pipeline (MuQ, translator, alignment, loss, BigVGAN, clamp) is reused unchanged from
TASK-15/16; `scripts/train_task18_heldout.py` wires these pieces to it.

Crop geometry is TASK-15's: 187 BigVGAN hops = 47,872 samples = 1.995 s at 24 kHz.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import random
from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np
import torch

SAMPLE_RATE = 24_000
GENRES = ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock")
EVAL_SETS = ("memorized", "train_unseen", "val", "test")
CONDITIONS = ("correct", "shuffled_across", "shuffled_time", "constant_zero", "average_mel")
CROP_FIELDS = [
    "crop_id", "role", "eval_set", "split", "track_id", "genre", "path", "checksum_sha256",
    "crop_start_sample", "crop_start_seconds", "crop_num_samples", "listening",
]
TRACK_FIELDS = ["split", "track_id", "genre", "path", "checksum_sha256", "duration_seconds", "resampled_num_samples"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- tracks -------------------------------------------------------------------------------

def select_tracks(rows: Sequence[dict], counts: dict[str, int], seed: int) -> list[dict]:
    """Deterministic, genre-balanced selection from the committed GTZAN split.

    rows: `gtzan_manifest_*.csv` rows (track_id, genre, relative_path, checksum, duration_seconds,
    corrupted, split). Every track whose checksum occurs more than once anywhere in the dataset is
    excluded, so checksum-identical audio can never sit in two splits (or twice in one). For each
    split, `counts[split] // 10` tracks per genre, plus one more for `counts[split] % 10` genres
    chosen by a seeded RNG -- for a held-out split, the genres the previous held-out split omitted
    are chosen first, so val + test cover all genres. Tracks within a genre are drawn by the same
    RNG from the sorted list.
    """
    checksum_counts = collections.Counter(r["checksum"] for r in rows)
    selected = []
    omitted_before: set[str] = set()  # genres an earlier held-out split left out
    for split, n in counts.items():
        rng = random.Random(f"task18-{seed}-{split}")
        eligible: dict[str, list[dict]] = {g: [] for g in GENRES}
        for r in rows:
            if (r["split"] == split and r["corrupted"] != "True" and checksum_counts[r["checksum"]] == 1
                    and float(r["duration_seconds"]) >= 29.0):
                eligible[r["genre"]].append(r)
        base, extra = divmod(n, len(GENRES))
        # Genres omitted by the previous held-out split come first, so val and test together
        # cover every genre whenever their sizes allow.
        forced = sorted(omitted_before)[:extra] if split != "train" else []
        rest = [g for g in sorted(GENRES) if g not in forced]
        extra_genres = set(forced) | set(rng.sample(rest, extra - len(forced)))
        if split != "train" and base == 0:
            omitted_before = set(GENRES) - extra_genres
        for g in GENRES:
            k = base + (g in extra_genres)
            pool = sorted(eligible[g], key=lambda r: r["track_id"])
            if len(pool) < k:
                raise ValueError(f"{split}/{g}: need {k} eligible tracks, have {len(pool)}")
            selected += [{**r, "split": split} for r in rng.sample(pool, k)]
    ids = [r["track_id"] for r in selected]
    assert len(ids) == len(set(ids)), "track selected twice"
    return sorted(selected, key=lambda r: (["train", "val", "test"].index(r["split"]), r["track_id"]))


def resampled_length(native_num_frames: int, native_sr: int, target_sr: int = SAMPLE_RATE) -> int:
    """Lower bound on the resampled length (soxr output is ceil-or-floor of the exact ratio)."""
    return math.floor(native_num_frames * target_sr / native_sr)


# --- crops --------------------------------------------------------------------------------

def allowed_intervals(max_start: int, reserved_starts: Sequence[int], crop_len: int) -> list[tuple[int, int]]:
    """Inclusive integer intervals of start samples s in [0, max_start] whose crop [s, s+crop_len)
    does not overlap any reserved crop [e, e+crop_len), i.e. s <= e - crop_len or s >= e + crop_len."""
    intervals = [(0, max_start)]
    for e in sorted(reserved_starts):
        lo, hi = e - crop_len + 1, e + crop_len - 1  # forbidden, inclusive
        nxt = []
        for a, b in intervals:
            if b < lo or a > hi:
                nxt.append((a, b))
                continue
            if a < lo:
                nxt.append((a, lo - 1))
            if b > hi:
                nxt.append((hi + 1, b))
        intervals = nxt
    return [(a, b) for a, b in intervals if a <= b]


def stratified_starts(intervals: list[tuple[int, int]], k: int, rng: random.Random) -> list[int]:
    """k start samples, one uniformly random draw in each of k equal-measure strata of the allowed
    set, so training crops spread across the whole track rather than clustering."""
    sizes = [b - a + 1 for a, b in intervals]
    total = sum(sizes)
    if total < k:
        raise ValueError(f"only {total} allowed starts for {k} crops")
    starts = []
    for i in range(k):
        u = min(int((i + rng.random()) / k * total), total - 1)
        for (a, _), size in zip(intervals, sizes, strict=True):
            if u < size:
                starts.append(a + u)
                break
            u -= size
    return starts


def build_crop_rows(tracks: list[dict], cfg: dict, seed: int) -> list[dict]:
    """All fixed crops, decided before training.

    - val/test tracks: `heldout_eval_offsets_seconds` (eval_set val/test).
    - train tracks: `train_unseen_offsets_seconds` (eval_set train_unseen) are reserved; the
      training pool (`train_pool_crops_per_track` stratified random starts) never overlaps them.
    - memorized: `memorized_pool_indices` of each training track's pool, evaluated as-is.
    - listening: first `listening_*` rows flagged (fixed before training)."""
    crop_len = cfg["num_mel_hops"] * cfg["hop_size"]
    margin = cfg.get("end_margin_samples", 2400)
    rows: list[dict] = []

    def row(t, role, eval_set, idx, start, listening=0):
        return {
            "crop_id": f"{t['track_id']}:{role}:{eval_set or 'pool'}:{idx}",
            "role": role, "eval_set": eval_set, "split": t["split"], "track_id": t["track_id"],
            "genre": t["genre"], "path": t["path"], "checksum_sha256": t["checksum_sha256"],
            "crop_start_sample": start, "crop_start_seconds": f"{start / SAMPLE_RATE:.6f}",
            "crop_num_samples": crop_len, "listening": listening,
        }

    for t in tracks:
        max_start = int(t["resampled_num_samples"]) - crop_len - margin
        if t["split"] in ("val", "test"):
            for i, sec in enumerate(cfg["heldout_eval_offsets_seconds"]):
                start = round(sec * SAMPLE_RATE)
                assert start <= max_start, (t["track_id"], sec)
                rows.append(row(t, "eval", t["split"], i, start))
            continue
        reserved = [round(s * SAMPLE_RATE) for s in cfg["train_unseen_offsets_seconds"]]
        for i, start in enumerate(reserved):
            assert start <= max_start, (t["track_id"], start)
            rows.append(row(t, "eval", "train_unseen", i, start))
        rng = random.Random(f"task18-{seed}-pool-{t['track_id']}")
        starts = stratified_starts(allowed_intervals(max_start, reserved, crop_len), cfg["train_pool_crops_per_track"], rng)
        for i, start in enumerate(starts):
            rows.append(row(t, "train_pool", "", i, start))
        for i in cfg["memorized_pool_indices"]:
            rows.append(row(t, "eval", "memorized", i, starts[i]))

    # Listening set: fixed before training, one crop per track at `listening_crop_index`
    # (val, test), plus the first `listening_train_tracks` training tracks' unseen and memorized crops.
    li = cfg["listening_crop_index"]
    train_ids = sorted({t["track_id"] for t in tracks if t["split"] == "train"},
                       key=lambda tid: (GENRES.index(tid.split(".")[0]), tid))
    chosen_train, seen_genres = [], set()
    for tid in train_ids:  # one per genre, in genre order
        g = tid.split(".")[0]
        if g not in seen_genres and len(chosen_train) < cfg["listening_train_tracks"]:
            chosen_train.append(tid)
            seen_genres.add(g)
    for r in rows:
        idx = int(r["crop_id"].rsplit(":", 1)[1])
        if r["eval_set"] in ("val", "test") and idx == li:
            r["listening"] = 1
        if r["eval_set"] in ("train_unseen", "memorized") and r["track_id"] in chosen_train and idx in (0, cfg["memorized_pool_indices"][0]):
            r["listening"] = 1
    return rows


def check_crop_rows(rows: list[dict]) -> dict:
    """Invariants: no track in two splits; unseen-crop windows never overlap a training-pool crop
    of the same track; memorized crops are exactly pool crops; every eval set non-empty."""
    split_of: dict[str, str] = {}
    checks: dict[str, int] = collections.Counter()
    for r in rows:
        if split_of.setdefault(r["track_id"], r["split"]) != r["split"]:
            raise ValueError(f"{r['track_id']} in two splits")
        checks[r["eval_set"] or "train_pool"] += 1
    pool = collections.defaultdict(list)
    for r in rows:
        if r["role"] == "train_pool":
            pool[r["track_id"]].append(int(r["crop_start_sample"]))
    for r in rows:
        s, n = int(r["crop_start_sample"]), int(r["crop_num_samples"])
        if r["eval_set"] == "train_unseen":
            bad = [p for p in pool[r["track_id"]] if p < s + n and s < p + n]
            if bad:
                raise ValueError(f"{r['crop_id']} overlaps training crops at {bad[:3]}")
        if r["eval_set"] == "memorized" and s not in pool[r["track_id"]]:
            raise ValueError(f"{r['crop_id']} is not a training-pool crop")
        if r["split"] in ("val", "test") and r["role"] == "train_pool":
            raise ValueError(f"{r['crop_id']}: held-out track in the training pool")
    for s in EVAL_SETS:
        if checks[s] == 0:
            raise ValueError(f"eval set {s} empty")
    return dict(checks)


# --- training sampler ---------------------------------------------------------------------

@dataclasses.dataclass
class TrackBalancedSampler:
    """Each batch draws `batch_size` tracks from a shuffled cycle over all training tracks
    (every track appears once per cycle), then one pool crop uniformly within each track. Exact,
    resumable state; per-track and per-crop exposure counts are kept for logging."""

    pool_by_track: dict[str, list[int]]  # track_id -> indices into the feature tensors
    batch_size: int
    rng: random.Random
    queue: list[str] = dataclasses.field(default_factory=list)
    track_exposure: dict[str, int] = dataclasses.field(default_factory=dict)
    crop_exposure: dict[int, int] = dataclasses.field(default_factory=dict)
    num_batches: int = 0

    def next_batch(self) -> list[int]:
        out = []
        for _ in range(self.batch_size):
            if not self.queue:
                self.queue = sorted(self.pool_by_track)
                self.rng.shuffle(self.queue)
            tid = self.queue.pop()
            idx = self.pool_by_track[tid][self.rng.randrange(len(self.pool_by_track[tid]))]
            self.track_exposure[tid] = self.track_exposure.get(tid, 0) + 1
            self.crop_exposure[idx] = self.crop_exposure.get(idx, 0) + 1
            out.append(idx)
        self.num_batches += 1
        return out

    def state_dict(self) -> dict:
        return {"rng_state": self.rng.getstate(), "queue": list(self.queue), "track_exposure": dict(self.track_exposure),
                "crop_exposure": dict(self.crop_exposure), "num_batches": self.num_batches}

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng_state"])
        self.queue = list(state["queue"])
        self.track_exposure = dict(state["track_exposure"])
        self.crop_exposure = dict(state["crop_exposure"])
        self.num_batches = state["num_batches"]

    def exposure_summary(self) -> dict[str, float]:
        tracks = [self.track_exposure.get(t, 0) for t in self.pool_by_track]
        n_crops = sum(len(v) for v in self.pool_by_track.values())
        return {
            "tracks_total": len(tracks), "tracks_visited": sum(c > 0 for c in tracks),
            "track_exposure_min": min(tracks), "track_exposure_max": max(tracks),
            "track_exposure_mean": float(np.mean(tracks)),
            "track_exposure_max_over_min": (max(tracks) / min(tracks)) if min(tracks) else float("inf"),
            "pool_crops_total": n_crops, "pool_crops_visited": len(self.crop_exposure),
            "pool_crop_coverage": len(self.crop_exposure) / n_crops,
        }


# --- conditioning controls ----------------------------------------------------------------

def shuffled_across_index(track_ids: Sequence[str]) -> list[int]:
    """Deterministic: crop i takes the MuQ features of the nearest following crop (cyclic) whose
    track differs, skipping by the largest per-track count so pairings spread across tracks."""
    n = len(track_ids)
    if len(set(track_ids)) < 2:
        raise ValueError("shuffled-across needs >= 2 tracks")
    step = max(collections.Counter(track_ids).values())
    out = []
    for i in range(n):
        for d in range(step, step + n):
            j = (i + d) % n
            if track_ids[j] != track_ids[i]:
                out.append(j)
                break
    return out


def shuffled_time_perms(n: int, num_frames: int, seed: int) -> torch.Tensor:
    """(n, num_frames) fixed per-crop permutations of MuQ frames."""
    return torch.stack([torch.randperm(num_frames, generator=torch.Generator().manual_seed(seed * 100_003 + i)) for i in range(n)])


def conditioned_features(condition: str, feats: torch.Tensor, across: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    """feats: (n, T_muq, C) for one eval set -> the translator input under `condition`.
    `average_mel` does not use the translator and is handled by the caller."""
    if condition == "correct":
        return feats
    if condition == "shuffled_across":
        return feats[across]
    if condition == "shuffled_time":
        return torch.gather(feats, 1, perms.to(feats.device).unsqueeze(-1).expand_as(feats))
    if condition == "constant_zero":
        return torch.zeros_like(feats)
    raise ValueError(condition)


# --- metrics ------------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mel_metrics(pred: torch.Tensor, target: torch.Tensor, floor: float) -> dict[str, float]:
    """pred/target: (num_mels, T) raw log-mels (pred unclamped, as in TASK-16's mel metrics)."""
    p, t = pred.detach().float().cpu(), target.detach().float().cpu()
    pc, tc = (p - floor).flatten(), (t - floor).flatten()
    return {
        "mel_l1": float((p - t).abs().mean()),
        "mel_cosine": float(torch.nn.functional.cosine_similarity(pc, tc, dim=0)),
        "mel_pearson": _pearson(p.numpy(), t.numpy()),
        "mel_variance_ratio": float((p.var(dim=-1, unbiased=False) / t.var(dim=-1, unbiased=False).clamp(min=1e-8)).mean()),
    }


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return x
    return np.convolve(x, np.ones(frames) / frames, mode="same")


def contour_metrics(reference: np.ndarray, candidate: np.ndarray, sr: int, onset_smoothing_ms: Sequence[float],
                    n_fft: int = 1024, hop: int = 256) -> dict[str, float]:
    """Phase-insensitive similarity of two equal-length waveforms: RMS (loudness) contour,
    spectral-centroid (brightness) contour, chroma (mean per-frame cosine), and onset-strength
    envelope at several smoothing timescales. Pearson r unless noted; NaN when a contour is flat."""
    ref, cand = np.asarray(reference, np.float32), np.asarray(candidate, np.float32)
    out = {
        "rms_corr": _pearson(librosa.feature.rms(y=ref, frame_length=n_fft, hop_length=hop)[0],
                             librosa.feature.rms(y=cand, frame_length=n_fft, hop_length=hop)[0]),
        "centroid_corr": _pearson(librosa.feature.spectral_centroid(y=ref, sr=sr, n_fft=n_fft, hop_length=hop)[0],
                                  librosa.feature.spectral_centroid(y=cand, sr=sr, n_fft=n_fft, hop_length=hop)[0]),
    }
    cr = librosa.feature.chroma_stft(y=ref, sr=sr, n_fft=n_fft * 4, hop_length=hop)
    cc = librosa.feature.chroma_stft(y=cand, sr=sr, n_fft=n_fft * 4, hop_length=hop)
    num = (cr * cc).sum(0)
    den = np.linalg.norm(cr, axis=0) * np.linalg.norm(cc, axis=0)
    valid = den > 1e-8
    out["chroma_cosine"] = float((num[valid] / den[valid]).mean()) if valid.any() else float("nan")
    o_r = librosa.onset.onset_strength(y=ref, sr=sr, hop_length=hop)
    o_c = librosa.onset.onset_strength(y=cand, sr=sr, hop_length=hop)
    for ms in onset_smoothing_ms:
        frames = max(1, round(ms / 1000 * sr / hop))
        out[f"onset_corr_{int(ms)}ms"] = _pearson(_smooth(o_r, frames), _smooth(o_c, frames))
    return out


# --- statistics ---------------------------------------------------------------------------

def cluster_bootstrap_delta(correct: Sequence[float], control: Sequence[float], clusters: Sequence[str],
                            lower_is_better: bool, seed: int = 0, num_bootstrap: int = 2000) -> dict[str, float]:
    """Paired advantage of `correct` over `control` (positive = correct better), with a 95% CI
    from resampling *tracks* (clusters) with replacement, since crops of one track are not
    independent. NaN pairs are dropped."""
    c, k = np.asarray(correct, float), np.asarray(control, float)
    delta = (k - c) if lower_is_better else (c - k)
    ok = ~np.isnan(delta)
    delta, cl = delta[ok], np.asarray(clusters)[ok]
    if delta.size == 0:
        return {"n": 0, "n_tracks": 0, "mean_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "proportion_crops_positive": float("nan"), "proportion_tracks_positive": float("nan")}
    tracks = sorted(set(cl))
    by_track = {t: delta[cl == t] for t in tracks}
    rng = np.random.default_rng(seed)
    means = np.empty(num_bootstrap)
    for b in range(num_bootstrap):
        pick = rng.integers(0, len(tracks), len(tracks))
        means[b] = np.concatenate([by_track[tracks[i]] for i in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(delta.size), "n_tracks": len(tracks), "mean_delta": float(delta.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "proportion_crops_positive": float((delta > 0).mean()),
        "proportion_tracks_positive": float(np.mean([by_track[t].mean() > 0 for t in tracks])),
    }


# --- pre-registered outcome rule ----------------------------------------------------------

OUTCOME_TEXT = {
    "UNDERFIT": "The translator does not fit even its own training crops; failure cannot be attributed to MuQ.",
    "ALL_FAIL": "The translator fits its training crops, but no held-out regime beats every control.",
    "TRACK_SPECIFIC_ONLY": "Unseen crops from known training tracks beat every control, but validation and/or test "
                           "tracks do not: track-specific generalization only.",
    "HELDOUT_COARSE": "Validation and test tracks beat every control on mel error, but stay well above BigVGAN's "
                      "round-trip error: held-out signal at the coarse level. Do not call this recognizable "
                      "reconstruction unless listening says so.",
    "HELDOUT_NEAR_ORACLE": "Validation and test tracks beat every control and approach BigVGAN's round-trip error. "
                           "Held-out track generalization if listening confirms recognizably source-related audio.",
}


def regime_status(summary: dict, rule: dict) -> dict:
    """summary: aggregate for one eval set (from the training script). Beats a control when the
    track-clustered 95% CI of the mel-L1 advantage is entirely above zero AND correct conditioning
    wins on at least `min_track_fraction_positive` of the tracks (a bootstrap over only 8 tracks
    gives optimistic intervals, so the CI alone is not enough)."""
    frac = rule.get("min_track_fraction_positive", 0.0)
    beats = {c: summary["paired_mel_l1"][c]["ci_low"] > 0 and summary["paired_mel_l1"][c]["proportion_tracks_positive"] >= frac
             for c in CONDITIONS if c != "correct"}
    correct = summary["conditions"]["correct"]["mel_l1"]["mean"]
    rt = summary["oracle_round_trip_mel_l1"]["mean"]
    return {"beats": beats, "beats_all": all(beats.values()), "mel_l1_over_round_trip": correct / rt}


def classify_outcome(aggregates: dict[str, dict], rule: dict) -> dict:
    """Pre-registered (configs/task18_heldout.yaml `outcome_rule`). Listening is recorded
    separately and is required before calling any audio recognizable."""
    st = {s: regime_status(aggregates[s], rule) for s in EVAL_SETS}
    mem = st["memorized"]
    if not mem["beats_all"] or mem["mel_l1_over_round_trip"] > rule["memorized_fit_ratio_max"]:
        code = "UNDERFIT"
    elif st["val"]["beats_all"] and st["test"]["beats_all"]:
        near = all(st[s]["mel_l1_over_round_trip"] <= rule["near_oracle_ratio_max"] for s in ("val", "test"))
        code = "HELDOUT_NEAR_ORACLE" if near else "HELDOUT_COARSE"
    elif st["train_unseen"]["beats_all"]:
        code = "TRACK_SPECIFIC_ONLY"
    else:
        code = "ALL_FAIL"
    return {"code": code, "text": OUTCOME_TEXT[code], "regimes": st, "listening_verdict": "pending",
            "note": "Automatic, mel-level classification. Recognizability needs the listening verdict; "
                    "memorized-crop results are never evidence of MuQ invertibility."}
