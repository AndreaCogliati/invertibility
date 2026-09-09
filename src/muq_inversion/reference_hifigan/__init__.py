"""Vendored, near-verbatim port of the official HiFi-GAN reference implementation.

Source: https://github.com/jik876/hifi-gan (commit: master, fetched 2026-09-08)
Paper: Kong, Kim & Bae, "HiFi-GAN: Generative Adversarial Networks for Efficient and High
Fidelity Speech Synthesis", NeurIPS 2020.
License: MIT (Copyright (c) 2020 Jungil Kong) -- see LICENSE_HIFIGAN.txt in this directory.

Used only by `scripts/train_reference_hifigan.py` (experiment/reference-hifigan branch) as a
faithfulness control: does the *unmodified*, published HiFi-GAN architecture/loss/optimizer
recipe reconstruct audio noticeably better than this project's own custom generator/
discriminators/losses, under the same tiny-overfit training budget? Not part of this project's
own decoder design (`models/generator.py`, `models/discriminators.py`) and not used by the main
MuQ pipeline. See notes/SCRATCHPAD.md ("New branch: experiment/reference-hifigan") for the full
fidelity plan, and notes/DECISIONS.md#reference-hifigan for the results.
"""
