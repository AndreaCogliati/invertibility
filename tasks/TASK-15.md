Create the next task for a small-scale MuQ → mel spectrogram → BigVGAN-v2 reconstruction experiment.

Goal: test the complete pipeline and determine whether a learned translator can memorize a tiny set of fixed audio crops. This is only a mechanical/memorization test, not a generalization experiment.

Pipeline:

audio
→ frozen MuQ last_hidden_state
→ learned translator
→ BigVGAN-v2 mel spectrogram
→ frozen BigVGAN-v2
→ reconstructed audio

Use the official NVIDIA BigVGAN implementation and pretrained Hugging Face model:

nvidia/bigvgan_v2_24khz_100band_256x

First:
1. Download/load the official BigVGAN code and checkpoint from Hugging Face using the supported NVIDIA/Hugging Face mechanism.
2. Inspect the model configuration and official mel preprocessing code.
3. Use exactly the mel representation expected by this checkpoint, including its sample rate, number of mel bands, FFT/window/hop parameters, frequency limits, scaling, and normalization. Do not independently choose or approximate these parameters.
4. Before training anything, verify the positive control:
   original waveform → official BigVGAN mel preprocessing → frozen BigVGAN-v2 → waveform.
   Save listening examples and quantitative metrics. Stop and diagnose if this oracle path does not reconstruct well.

Then implement the learned path:

audio
→ frozen MuQ last_hidden_state
→ translator
→ BigVGAN-compatible mel
→ frozen BigVGAN-v2
→ audio.

Experiment:
- Start with N=2 fixed deterministic 2-second crops.
- Train only the translator; MuQ and BigVGAN remain frozen.
- Choose the simplest reasonable temporal translator that handles the MuQ-to-BigVGAN-mel frame-rate conversion. Record the exact geometry and parameter count.
- Monitor training with Weights & Biases.
- Compare target vs predicted mel directly.
- Decode both through exactly the same frozen BigVGAN model.
- Save, for each crop:
  original,
  BigVGAN oracle,
  predicted-mel reconstruction.
- Include predicted-vs-target mel plots and appropriate mel/spectral/audio metrics.
- Include a shuffled-across MuQ control to verify that successful reconstruction depends on the correct MuQ input.

Decision:
- If N=2 cannot closely memorize the target mel and approach the BigVGAN oracle perceptually, stop and diagnose the pipeline.
- If N=2 succeeds, repeat at N=8.
- Do not scale beyond N=8, test held-out generalization, change MuQ layers, tune BigVGAN, or perform an architecture/loss sweep.

Keep this experiment and its artifacts clearly separate from the EnCodec experiments.

Do not commit unless explicitly requested.

At the end, report only whether the MuQ → BigVGAN-mel → BigVGAN pipeline is mechanically viable in the small-N memorization regime. Do not infer generalization or MuQ invertibility from success.
