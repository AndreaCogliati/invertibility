Implement the two pipeline corrections identified by the TASK-15 code audit, then rerun only the N=2 smoke/memorization test to verify that the successful result is preserved.

1. Fix MuQ → BigVGAN mel temporal alignment
- Replace generic `F.interpolate(..., align_corners=False)` length resizing with explicit interpolation based on the physical frame centers used by the actual implementations.
- Derive the MuQ token centers and BigVGAN mel-frame centers from the code/model configuration rather than hard-coding undocumented assumptions.
- Map each BigVGAN mel frame to the corresponding time in the MuQ sequence and interpolate MuQ features there.
- Add tests that verify the mapping, endpoints and tensor shapes.

2. Enforce the BigVGAN mel floor at decoding
- Preserve the translator's raw output for training and diagnostic metrics.
- Immediately before frozen BigVGAN inference, clamp predicted mel values to the same minimum valid log-mel value used by BigVGAN's official preprocessing.
- Derive that floor from the implementation/configuration where possible.
- Report how many predicted bins are affected by the clamp.

Then rerun TASK-15 only at N=2 using the corrected pipeline:
- same two deterministic crops;
- same translator, loss and optimizer;
- same fixed training budget;
- same seed;
- same BigVGAN checkpoint;
- same best-checkpoint policy;
- same shuffled-across control.

Compare the corrected run directly with the previous N=2 result:
- target/predicted mel error;
- predicted-vs-BigVGAN-oracle audio metrics;
- variance or other existing diagnostics;
- shuffled-across control;
- number/fraction of mel bins clamped;
- listening files: original, BigVGAN oracle, corrected prediction.

Do not run N=8 or any generalization experiment.

Update the pipeline-audit Markdown diagram so it reflects the corrected time-coordinate interpolation and decode-time mel clamp. Clearly distinguish raw predicted mel from the clamped mel sent to BigVGAN.

Conclude only whether these corrections materially change the previous TASK-15 N=2 pipeline result. Do not reinterpret the earlier experiment or make claims about generalization/invertibility.

Run relevant tests and lint. Do not commit unless explicitly requested.