# Local-Global Multi-Instance Network

This experiment trains one shared image backbone over five views of every
sample: one complete image and four zoomed quadrant regions. A learned
attention module fuses intermediate features, so this is not probability
averaging or conventional test-time multi-crop.

It uses only image pixels, trains from scratch, preserves the fixed
train/valid/test splits, and remains under the 5M parameter limit. Geographic
metadata is loaded by the shared dataset code for compatibility but is never
passed to the model.

The training set always contains 7,560 unique images. Augmentation is generated
online: 100 epochs produce 756,000 augmented sample presentations. Each sample
contains five views, corresponding to 3,780,000 view tensors processed over the
full run; these are not additional independent images.

## L4 command

```bash
python -u -m experiments_local_global_mil.train \
  --data-root /teamspace/studios/this_studio/dataset \
  --epochs 100 \
  --batch-size 8 \
  --accumulation-steps 4 \
  --num-workers 0 \
  2>&1 | tee experiments_local_global_mil/logs/train_l4.log
```

Five views increase activation memory and compute, but the shared backbone is
counted only once. Run the default batch 8 first on a 24GB L4.

Every epoch reports both raw-model and warm-started EMA validation accuracy.
The better validation checkpoint is retained. `checkpoints/last_state.pt` is
written every epoch and can be resumed with `--resume PATH`.
