# Biomedical Image Analysis Pipeline

A reproducible starter project for image classification on the **PathMNIST** histopathology dataset from MedMNIST.

## What it does
- Downloads a public, small biomedical image benchmark through MedMNIST.
- Applies consistent image transforms and creates train/validation/test loaders.
- Trains a compact CNN baseline in PyTorch.
- Reports accuracy, macro-F1, a classification report, and a confusion matrix.
- Saves the trained model and evaluation artefacts under `outputs/`.

> This is a learning/demo pipeline, not a clinical diagnostic tool. Benchmark results are not evidence of clinical performance.

## Quick start
Use Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train --epochs 3
```

The first run downloads PathMNIST and may take a little while. For a quick smoke test, use `--epochs 1`.

## Repository structure
```text
src/
  data.py       dataset and data-loader setup
  model.py      compact CNN
  train.py      training, evaluation, and artefact saving
```

## Reproducibility
- Fixes Python and PyTorch random seeds.
- Uses explicit train/test splits provided by MedMNIST.
- Records the run configuration and test metrics as JSON.
- Keeps data and generated artefacts out of Git.

## Baseline Experiment Results

**Dataset:** PathMNIST (MedMNIST)  
**Model:** Custom convolutional neural network (CNN)  
**Training device:** Apple Metal Performance Shaders (MPS)  
**Random seed:** 42  
**Training duration:** 3 epochs

| Metric | Result |
|---|---:|
| Test accuracy | 81.35% |
| Test macro-F1 | 0.7650 |
| Test loss | 0.7429 |
| Best validation macro-F1 | 0.8067 |

Training loss decreased from 1.0439 to 0.5998 over three epochs. Validation loss decreased from 0.7933 to 0.5161, while validation macro-F1 increased from 0.7070 to 0.8067.

### Limitations and next steps

This is an initial baseline, not a clinically validated model. Further work could include longer training, learning-rate tuning, class-wise error analysis, a confusion-matrix visualisation, and comparison with transfer learning. Results should be reproduced across multiple seeds before drawing strong conclusions.
