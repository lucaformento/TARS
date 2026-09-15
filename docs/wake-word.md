# Hey TARS wake-word decision

The Raspberry Pi prototype now uses a custom **Hey TARS** openWakeWord model.
The model is a practical project baseline rather than a research-grade detector.
Its known tradeoff is accepted: phrases such as “hey stars” or “hey cars” can
occasionally wake TARS.

## What was tested

The first candidate reported 69.3% recall on synthetic target clips, a 1.1%
positive fraction across pooled synthetic negatives, and zero positives across
56,250 development background windows at threshold 0.5. Synthetic validation
also influenced model selection, so these are screening figures rather than an
independent accuracy claim.

On the Pi, all nine clean and correctly prompted trials spoken by the builder
crossed threshold 0.5. Clean false triggers were observed for “hey stars” and
“hey Jarvis.” A “hey cars” trial also triggered but had an input overflow.
Several unexplained activations occurred during a later acknowledgment test.

Three more tightly controlled training attempts were evaluated:

| Attempt | Best result | Outcome |
| :--- | ---: | :--- |
| v2 | 75.0% target recall before all safety gates | Phrase and background gates never passed together |
| v2.1 | 52.5% strict-safe recall | Rejected |
| v2.2 | 53.0% strict-safe recall | Rejected; synthetic retraining stopped |

The v2.2 hard-example pass improved separation from ordinary background noise,
but rare close-phrase examples still overlapped with valid target examples.
Raising the threshold enough to reject those outliers caused too many missed
wakes.

## Runtime policy

The application keeps the original candidate at threshold 0.5 because it has
the strongest evidence with the intended speaker. It also:

- uses the model's matching mel-spectrogram and embedding ONNX files;
- enables openWakeWord voice-activity gating at 0.5 to reduce non-speech
  activations; and
- waits two seconds after returning to sleep before accepting another wake.

A consecutive-frame requirement remains an optional future tuning knob. It was
not enabled because the saved trials contain peak scores rather than full frame
histories, so its effect on valid wakes is unknown.

The candidate model SHA-256 is
`4d7e339df7096a38b279b5322f27c51de815fd20d3dc00f9e5a689794aa03bed`.
Model and feature files are provisioned locally and excluded from Git.

Revisit training only if false or missed wakes materially interfere with normal
use. Future evaluation should use continuous real-room audio rather than another
synthetic-only training cycle.
