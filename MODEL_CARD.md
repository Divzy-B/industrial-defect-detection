# Model card

## Intended use

This service is a starting point for visual inspection of manufactured surfaces.
It detects localized visual anomalies in still images and returns labeled boxes
and confidence scores. It is intended to support—not replace—trained quality
inspectors and established safety controls.

## Model and classes

The repository intentionally ships **without weights**. Supply a versioned
YOLOv8 `.pt` model or an ONNX export through `DEFECT_MODEL_PATH`. The default
labels are `crazing`, `inclusion`, `patches`, `pitted_surface`,
`rolled-in_scale`, and `scratches`; replace them when using another dataset.

## Evaluation requirements

Evaluate a candidate model on a held-out sample from every line, camera,
material grade, lighting regime, and defect severity where it will operate.
Report per-class precision, recall, mAP, false-negative rate, latency, and
calibration. Set acceptance thresholds together with process owners.

## Limitations and operational safeguards

- Confidence is not a probability of product fitness or safety.
- Domain shift (camera position, glare, contamination, texture, or new lots)
  can sharply reduce accuracy.
- False negatives can be costly. Route low-confidence or no-detection cases
  according to a documented human-review policy.
- Record model version, threshold, image source, and reviewer decision for
  traceability.
- Revalidate after retraining, lens/lighting changes, or software upgrades.

