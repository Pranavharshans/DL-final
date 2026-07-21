# Model Results — Image Country Classification

V1: Trained on RTX 2060 12GB, 256×256, 50 epochs, batch_size=32, no augmentation.
V2: Same hardware, 256×256, 50 epochs, batch_size=32, augmentation + label smoothing + warmup.
Models ranked by valid accuracy. V2 models marked with † use validation only (no test set eval).

| Rank | Model | Params | Train Acc | Valid Acc | Test Acc | Time |
|------|-------|--------|-----------|-----------|----------|------|
| 1 | 24_HRNet-Lite † | 4,963,347 | — | 0.8042 | — | 4443s |
| 2 | 25_CoAtNet-Micro † | 2,000,623 | — | 0.8042 | — | 3578s |
| 3 | 27_AntiAliased-ResNet-SK † | 4,511,875 | — | 0.8000 | — | 4687s |
| 4 | 30_MaxViT-Micro † | 3,713,891 | — | 0.7921 | — | 4541s |
| 5 | 22_Res2Net-ECA-GeM † | 1,020,413 | — | 0.7898 | — | 3894s |
| 6 | 23_GeoAuxiliary-ImageOnly † | 1,112,488 | — | 0.7884 | — | 3813s |
| 7 | 28_SpatialFrequency-Net † | 3,750,451 | — | 0.7838 | — | 3144s |
| 8 | 26_InceptionResNet-Lite † | 1,909,107 | — | 0.7796 | — | 4024s |
| 9 | 19_RepVGG-Style | 2,409,618 | 1.0000 | 0.7708 | 0.7500 | 1175s |
| 10 | 02_CustomCNN-Medium | 4,417,378 | 0.9810 | 0.7755 | 0.7630 | 2300s |
| 11 | 01_CustomCNN-Small | 1,578,802 | 0.9836 | 0.7648 | 0.7593 | 706s |
| 12 | 03_CustomCNN-Large | 2,739,922 | 0.9541 | 0.7718 | 0.7620 | 2584s |
| 13 | 29_CountryMetric-Net † | 4,588,193 | — | 0.7616 | — | 3539s |
| 14 | 10_MultiScale-CNN | 1,110,930 | 0.9454 | 0.7537 | 0.7278 | 1458s |
| 15 | 12_Wide-ResNet | 2,762,546 | 0.9306 | 0.7537 | 0.7454 | 6244s |
| 16 | 04_MiniResNet | 2,762,546 | 0.9044 | 0.7435 | 0.7481 | 13450s |
| 17 | 08_CBAM-Net | 2,784,638 | 0.9985 | 0.7398 | 0.7296 | 7765s |
| 18 | 11_DenseNet-Style | 786,514 | 0.9980 | 0.7394 | 0.7296 | 1072s |
| 19 | 21_MultiView-CNN † | 3,432,788 | — | 0.7338 | — | 4809s |
| 20 | 07_SE-ResNet | 2,784,050 | 0.9934 | 0.7171 | 0.6963 | 6803s |
| 21 | 13_CNN+SelfAttention | 2,828,594 | 1.0000 | 0.7116 | 0.6843 | 946s |
| 22 | 14_CNN+Transformer | 2,795,762 | 1.0000 | 0.7116 | 0.6870 | 1138s |
| 23 | 09_ConvNeXt-Micro | 2,888,674 | 1.0000 | 0.6935 | 0.6574 | 4000s |
| 24 | 05_MobileNet-Style | 1,357,138 | 1.0000 | 0.6782 | 0.6481 | 6374s |
| 25 | 31_CompactNet-XL | 4,663,954 | 1.0000 | 0.6750 | 0.6667 | 2023s |
| 26 | 18_EfficientNet-Style | 2,480,514 | 1.0000 | 0.6593 | 0.6269 | 1209s |
| 27 | 20_CompactNet | 277,619 | 1.0000 | 0.6208 | 0.6037 | 969s |
| 28 | 06_ShuffleNet-Style | 1,091,666 | 1.0000 | 0.5519 | 0.5296 | 1005s |
| 29 | 17_MultiTask-Geo | 2,779,124 | 0.8511 | 0.4583 | 0.4296 | 927s |
| 30 | 15_GeoDualBranch | 2,859,250 | 0.0587 | 0.0597 | 0.0593 | 918s |
| 31 | 16_GeoFiLM | 2,250,418 | 0.0556 | 0.0556 | 0.0556 | 967s |

**Key findings:**
- V2 models (21-30) dominate the top 8 spots thanks to augmentation, label smoothing, and warmup
- HRNet-Lite and CoAtNet-Micro tie for #1 at 80.42% valid accuracy
- V1 best (no augmentation): CustomCNN-Medium at 76.3% test / 77.6% valid
- Geo models (15-17) proved coordinate cheating — fixed with neutral mean during eval
- CompactNet-XL (31) shows depthwise+SE architecture needs augmentation to compete
