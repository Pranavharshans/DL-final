# Model Results — Image Country Classification

All models trained on RTX 2060 12GB, 512×512 images, 50 epochs, batch_size=32.

| Rank | Model | Params | Train Acc | Valid Acc | Test Acc | Time |
|------|-------|--------|-----------|-----------|----------|------|
| 1 | 01_CustomCNN-Small | 1,578,802 | 0.9836 | 0.7648 | 0.7593 | 706s |
| 2 | 02_CustomCNN-Medium | 4,417,378 | 0.9810 | 0.7755 | 0.7630 | 2300s |
| 3 | 03_CustomCNN-Large | 2,739,922 | 0.9541 | 0.7718 | 0.7620 | 2584s |
