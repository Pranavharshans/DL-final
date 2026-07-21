# CoAtNet-Large-4M Size-Only Control

This experiment isolates model capacity from augmentation and inference changes.
It uses the exact 4,252,785-parameter architecture from
`experiments_coatnet512_large`, but otherwise matches the current 2M baseline:

- 512x512 input and cache
- Original `GeoAugmentV1`
- AdamW, learning rate `1e-3`, weight decay `1e-4`
- Label smoothing `0.08`, three warmup epochs, cosine decay
- Seed 42, effective batch size 32, 50 epochs
- Single-crop validation and test inference
- No coordinates, external data, pretrained weights, or multi-crop

For a clean capacity comparison, keep micro-batch 16 and accumulation 2 because
that is how the current 2M L4 run was trained. Batch 32 would change BatchNorm
statistics and introduce another experimental variable.

```bash
cd ~/DL-final
git pull origin main
source .venv/bin/activate

python -u -m experiments_coatnet512_large_control.train \
  --data-root /teamspace/studios/this_studio/dataset \
  --epochs 50 \
  --batch-size 16 \
  --accumulation-steps 2 \
  --num-workers 0 \
  2>&1 | tee experiments_coatnet512_large_control/logs/train_l4.log
```

Afterward compare:

1. 2M baseline result: capacity baseline
2. This result: capacity effect only
3. Large multi-crop result: safe-augmentation and inference effect
