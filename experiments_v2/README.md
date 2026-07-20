# V2 Image-Only Geolocation Experiments

This directory contains ten new architectures trained from scratch under the
assignment's two constraints:

1. The complete submitted model has at most 5,000,000 trainable parameters.
2. Training uses only the provided images and metadata; no pretrained weights
   or external datasets are used.

Every model's normal `forward(image)` path is image-only. `GeoAuxiliaryNet`
uses the provided coordinates only as auxiliary training targets and never as
model inputs.

## Architectures

| ID | Architecture |
|---|---|
| 21 | Multi-view shared-backbone CNN |
| 22 | Res2Net-Lite + ECA + GeM |
| 23 | Image-only network with removable geographic auxiliary heads |
| 24 | HRNet-Lite |
| 25 | CoAtNet-Micro |
| 26 | Inception-ResNet-Lite |
| 27 | Anti-aliased ResNet-D + selective kernels |
| 28 | Spatial-frequency dual-branch network |
| 29 | Country-prototype metric network with ArcFace |
| 30 | MaxViT-Micro |

## Data policy

- Existing `train/`, `valid/`, and `test/` folders are read without alteration.
- Training images are cached as 288x288 uint8 tensors and receive online
  `GeoAugmentV1` transformations before becoming 256x256 inputs.
- Validation and test images receive deterministic resizing and normalization.
- Mean and standard deviation are computed from the provided training split.
- There are no flips or rotations that could corrupt driving-side, text, or
  road-layout clues.

## Verification

From the repository root:

```bash
pip install -r experiments_v2/requirements.txt
python -m experiments_v2.verify
python -m unittest discover -s experiments_v2/tests
```

These commands enforce split isolation, image-only forward passes, valid output
shapes, and the 5-million-parameter ceiling.

## Training

Train all ten models for the default 50 epochs:

```bash
python -m experiments_v2.run_all
```

Train one model:

```bash
python -m experiments_v2.run_all --model 22_Res2Net-ECA-GeM
```

Resume after an interruption:

```bash
python -m experiments_v2.run_all --resume-results
```

The runner evaluates and ranks models using validation accuracy only. It does
not read the test loader during model selection.

## Locked test evaluation

After selecting and freezing a winner from validation results:

```bash
python -m experiments_v2.evaluate experiments_v2/checkpoints/22_Res2Net-ECA-GeM.pt
```
