# CoAtNet-Large-4M at 512x512

This isolated experiment expands the successful CoAtNet-Micro to exactly
4,252,785 trainable parameters. It trains from scratch using only supplied
images and labels. Coordinates are never passed to the model.

## What is different

- Wider convolution stages and three late spatial-attention blocks
- Conservative augmentation: mild crop, color, blur, and erasing
- No flips, rotation, perspective distortion, external data, or pretrained weights
- Best checkpoint selected using ordinary validation accuracy
- Final six-view inference: full image plus center and four corner crops
- Test split evaluated exactly once, after all training and model selection

Multi-crop does not create or retrieve data. It takes six deterministic views
from each supplied image, evaluates them sequentially, and averages their class
probabilities. Sequential evaluation keeps peak GPU memory near single-crop
evaluation, although final evaluation takes approximately six times longer.

## L4 24 GB commands

```bash
cd ~/DL-final
git pull origin main
source .venv/bin/activate
pip install -r experiments_coatnet512_large/requirements.txt

python -m experiments_coatnet512_large.memory_check --batch-sizes 4 8 12 16

python -u -m experiments_coatnet512_large.train \
  --data-root /teamspace/studios/this_studio/dataset \
  --epochs 50 \
  --batch-size 8 \
  --accumulation-steps 4 \
  --num-workers 0 \
  2>&1 | tee experiments_coatnet512_large/logs/train_l4.log
```

If the memory check leaves at least 15% headroom at batch 16, use batch 16 and
accumulation 2 instead. Both configurations have effective batch size 32.

At completion, inspect:

```bash
cat experiments_coatnet512_large/results.md
```
