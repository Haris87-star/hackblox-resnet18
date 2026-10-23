# HackBlox 2026: Data-Centric ResNet-18

This repository is prepared for the fixed-architecture, six-class challenge. The model boundary is deliberately strict: `torchvision.models.resnet18(weights=None)` is the only classifier architecture, and no pretrained weights or external embeddings are used.

## Expected data layout

```text
data/
  train/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
    undefined/
  val/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
  test/
```

The starter audit can be run with:

```powershell
python -m src.train
```

It writes `outputs/dataset_audit.json`. The organizer kit uses `train/` for the
600 labeled images plus `undefined/` for the 6,000-image pool, `val/` for the
fixed validation set, and `test/` for inference.

## Integrity guardrails

- Maximum effective training set: 3,000 samples.
- Six output classes in the official order.
- Scratch initialization only.
- No fabricated metrics or 3LC artifacts.

## 24-hour execution plan

1. **Hour 0-1: audit.** Copy the organizer starter kit into `data/`, verify the
  six class counts, inspect image dimensions, and identify the Kaggle ID and
  prediction columns from the supplied sample submission.
2. **Hour 1-4: baseline.** Train the scratch ResNet-18 with a stratified split,
  save the seed/config/checkpoint, and export accuracy, per-class metrics,
  confusion matrix, and wrong predictions.
3. **Hour 4-7: inspect.** Load predictions and embeddings into 3LC. Focus on
  high-loss samples and the actual confusion pairs, especially glacier/mountain,
  buildings/street, and sea/mountain only when the measured confusion supports it.
4. **Hour 7-14: label deliberately.** Rank unlabeled images with uncertainty,
  embedding diversity, class balance, and boundary relevance. Label a small first
  batch, record its reason, retrain, and compare against a random-sampling control.
5. **Hour 14-19: consolidate.** Add only batches that improve repeated validation
  or repair a documented class gap. Keep the effective weighted sample count at
  or below 3,000.
6. **Hour 19-24: freeze.** Select the best defensible checkpoint, generate and
  validate the Kaggle submission, export 3LC tables/screenshots, and write the
  measured results and failure analysis.

## What wins this challenge

The strongest submission is not the one with the most code. It is the one that
can show a clean chain of evidence: baseline failure, a specific data problem,
an explicit sample-selection rule, human labels added for that reason, measured
improvement, and a reproducible final submission. Never report a metric or 3LC
artifact until it exists on disk.

## Current blocker

The repository does not yet contain the organizer starter kit, the 600 labeled
images, the 6,000-image pool, or Kaggle files. Once those are copied into the
workspace, run `\.venv\Scripts\python.exe run_pipeline.py` first. Do not guess
the submission schema or begin labeling until the audit identifies the actual
file layout.