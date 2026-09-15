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

### v2.2 diagnostic audit

The rejected v2.2 checkpoint reaches 73.0% development target recall when its
raw scores are evaluated directly at threshold 0.5. That is a modest increase
over the first candidate's 69.3% synthetic figure, but it does not satisfy the
phrase gates: at the same threshold, several negative groups report positive
fractions between 2% and 8%, including `hey stars`, `hey cars`, `cars`, and
`guitars`.

The strict v2.2 threshold was selected from background and per-phrase negative
scores. It happened to fall near the median target score because the target and
negative distributions overlap near 1.0; the target median itself does not set
the threshold. Consequently, the roughly 53% strict-safe recall remains useful
evidence that the configured gates cannot be met by that checkpoint.

The three saved background scores above 0.5 were isolated single-window peaks.
[`openWakeWord` 0.4.0 supports a `patience` option](https://github.com/dscripka/openWakeWord/blob/v0.4.0/openwakeword/model.py#L142-L158)
that can require consecutive positive windows, but its own API notes that this
may lower true-positive rate.
The diagnostics retain only per-clip target and phrase-negative peaks, so they
cannot establish two-window recall or phrase rejection. A persistence rule is
therefore an optional Pi experiment, not a validated 73% deployment result.

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
Its deployed feature models match the files used by training:

- mel-spectrogram: `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f`
- embedding: `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f`

The application passes both paths explicitly instead of relying on the
openWakeWord package defaults. Model and feature files are provisioned locally
and excluded from Git.

Revisit training only if false or missed wakes materially interfere with normal
use. Future evaluation should use continuous real-room audio rather than another
synthetic-only training cycle.
