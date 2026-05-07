# Instance Segmentation Classification Application

Run instance segmentation first, then crop the detected instances and run an ImageNet classifier on each crop.

```bash
hailo-seg-classify
```

The default model order is:

1. Instance segmentation model, for example `yolov5m_seg`
2. Classification model, for example `fastvit_sa12`

To override both models, repeat `--hef-path` in that order:

```bash
hailo-seg-classify --hef-path yolov8m_seg --hef-path resnet_v1_18
```

Mobilenet ImageNet classifiers are also supported:

```bash
hailo-seg-classify --hef-path yolov8s_seg --hef-path mobilenet_v2_1.0
```

The app prints each segmentation detection and, once the second stage produces a result, the top classification label and confidence. With `--use-frame`, it also overlays instance masks and the classification label on the displayed frame.

By default, the app classifies up to eight detected-object crops per frame. Lower this for heavier classification models or slower devices:

```bash
hailo-seg-classify --max-classification-crops 1 --classification-batch-size 1
```

To run without object tracking:

```bash
hailo-seg-classify --disable-tracking
```

For additional options:

```bash
hailo-seg-classify --help
```
