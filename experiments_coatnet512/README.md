# CoAtNet-Micro at 512x512

This experiment trains the existing 2,000,623-parameter CoAtNet-Micro from
scratch using 512x512 inputs. It uses only the provided images and labels.
Latitude and longitude are loaded by the shared dataset class but are never
passed to the model.

The validation split selects the best checkpoint. The internal test split is
evaluated exactly once after training has finished.

## Hardware

The recommended starting configurations are:

| GPU memory | Micro-batch | Accumulation | Effective batch |
|---:|---:|---:|---:|
| 12 GB minimum | 8 | 4 | 32 |
| 16 GB comfortable | 8 | 4 | 32 |
| 24 GB preferred | 16 | 2 | 32 |

The loader pre-caches all 10,800 images as 512x512 uint8 tensors. Allow about
8.5 GB for image tensors and use a VM with at least 24 GB of system RAM; 32 GB
is preferred. Keep `--num-workers 0` so worker processes do not copy the cache.

Actual CUDA consumption depends on the PyTorch/CUDA build. Run the included
memory probe before training and keep at least 15% headroom.

## Clone and install on the VM

```bash
git clone --branch main --single-branch https://github.com/Pranavharshans/DL-final.git
cd DL-final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r experiments_coatnet512/requirements.txt
```

Dataset JPEGs are intentionally excluded from Git. The repository must contain:

```text
train/labels.csv + 7,560 JPEGs
valid/labels.csv + 2,160 JPEGs
test/labels.csv  + 1,080 JPEGs
```

If the VM does not already contain them, upload the split from the local
machine after cloning:

```bash
rsync -av --progress train/ VM_USER@VM_HOST:/workspace/DL-final/train/
rsync -av --progress valid/ VM_USER@VM_HOST:/workspace/DL-final/valid/
rsync -av --progress test/ VM_USER@VM_HOST:/workspace/DL-final/test/
```

Verify the counts on the VM:

```bash
find train -maxdepth 1 -name '*.jpg' | wc -l
find valid -maxdepth 1 -name '*.jpg' | wc -l
find test -maxdepth 1 -name '*.jpg' | wc -l
```

Expected output is `7560`, `2160`, and `1080`.

## Check GPU memory

```bash
python -m experiments_coatnet512.memory_check --batch-sizes 4 8 12 16 24 32
```

Choose the largest passing batch with at least 15% memory headroom. Adjust
accumulation so `batch-size x accumulation-steps = 32`.

## Run training

For a 12 GB GPU:

```bash
mkdir -p experiments_coatnet512/logs
python -m experiments_coatnet512.train \
  --data-root /workspace/DL-final \
  --epochs 50 \
  --batch-size 8 \
  --accumulation-steps 4 \
  --num-workers 0 \
  2>&1 | tee experiments_coatnet512/logs/train.log
```

For a 24 GB GPU:

```bash
python -m experiments_coatnet512.train \
  --data-root /workspace/DL-final \
  --epochs 50 \
  --batch-size 16 \
  --accumulation-steps 2 \
  --num-workers 0 \
  2>&1 | tee experiments_coatnet512/logs/train.log
```

If the repository is cloned somewhere else, replace `/workspace/DL-final` with
the absolute clone path.

## Outputs

At completion, the script prints and saves:

- Best validation epoch and accuracy
- Deterministic train, validation, and test loss/accuracy
- Per-country validation and test accuracy
- Complete test confusion matrix
- Training time
- Peak CUDA allocated and reserved memory
- Model parameter count
- `experiments_coatnet512/checkpoints/coatnet_micro_512_best.pt`
- `experiments_coatnet512/results.json`
- `experiments_coatnet512/results.md`

The checkpoint is Git-ignored because it contains binary weights. Copy it to
persistent storage before terminating an ephemeral VM.
