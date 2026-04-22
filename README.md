# Grad-CAM on CIFAR-10

This project is a small experiment on model shortcuts and visual explanations.

The notebook trains two CNNs on CIFAR-10:

- a `clean` model trained on the original images
- a `biased` model trained on images with a label-dependent color patch

After training, the project compares both models with Grad-CAM to show how shortcut learning changes what the network focuses on.

## What is inside

- `grad_cam_cifar10.ipynb` — the full pipeline: data loading, patch generation, training, evaluation, Grad-CAM, and confidence-drop analysis
- `data/cifar-10-batches-py` — expected location of the raw CIFAR-10 batch files
- `artifacts/` — saved checkpoints for the clean and biased models

## How it works

The biased version of the dataset adds a small color patch to each image. The patch color depends on the class label, so the model can learn an artificial shortcut instead of relying on object features.

The notebook then:

1. trains or loads both models
2. evaluates them on clean and patched test sets
3. visualizes Grad-CAM heatmaps side by side
4. measures confidence drop after masking high-activation regions

## Setup

Install the main dependencies:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

Download CIFAR-10 in the original Python batch format and place it here:

```text
data/cifar-10-batches-py
```

## Run

Open the notebook and run the cells in order:

```bash
jupyter notebook grad_cam_cifar10.ipynb
```

The notebook can reuse checkpoints from `artifacts/` or retrain the models from scratch.

## Notes

- Default training uses 12 epochs.
- The synthetic shortcut patch is `8x8` by default.
- Grad-CAM maps are naturally coarse because CIFAR-10 images are only `32x32`.

## Expected takeaway

The clean model usually stays more reliable on clean images, while the biased model tends to focus more on the injected patch. The Grad-CAM comparison makes that shortcut behavior easy to see.
