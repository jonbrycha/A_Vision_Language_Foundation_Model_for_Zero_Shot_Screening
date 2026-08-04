# A Vision-Language Foundation Model for Zero-Shot Screening of Oral Herpetic and Periodontal Lesions

This repository contains the HiCPG evaluation stack for frozen BiomedCLIP encoders. It scores clinical photographs against anatomical, morphological, and differential-diagnostic prompt groups, combines group scores with calibrated weights, and produces split-conformal prediction sets.

## Installation

The reference environment uses Python 3.10, PyTorch 2.1.0, CUDA 11.8, and an NVIDIA A100 80 GB GPU.

```bash
conda env create -f environment.yml
conda activate hicpg
pip install -e .
```

## Data

Verified public source pages and access constraints are listed in `dataset_links.txt`. Prepare a manifest with one row per image:

```text
path,label,dataset,patient_id
images/example.jpg,herpes_simplex,bdj,case_001
```

```bash
python -m hicpg.data.prepare --manifest raw.csv --output data/manifest.csv
```

The software does not redistribute clinical images. Dataset terms remain controlling.

## Scoring

```bash
python -m hicpg.cli score --config configs/main.yaml --manifest data/manifest.csv --output outputs/scores.csv
```

Text embeddings are cached after the first pass. Image embeddings are computed once per image. The reported setup uses 224 by 224 center crops, ImageNet normalization, temperature 0.02, six templates per level, and level weights 0.2, 0.5, and 0.3.

## Calibration

```bash
python -m hicpg.cli calibrate --scores outputs/calibration.csv --alpha 0.10 --output outputs/calibrator.json
python -m hicpg.cli evaluate --scores outputs/evaluation.csv --calibrator outputs/calibrator.json --output outputs/metrics.json
```

The calibration partition must remain disjoint from model selection and evaluation. When no official split exists, use a stratified 80/20 calibration/evaluation split. The expected aggregate reference values are 62.4% mean balanced accuracy, 0.841 AUROC for herpes-versus-aphthous differentiation, 91.3% conformal coverage, and mean prediction-set size 2.52.

## Compute budget

Reference inference was measured with PyTorch 2.1 on one NVIDIA A100 80 GB GPU. Batch-one latency is approximately 50 ms per image with uncached prompts and 14.8 ms with cached text embeddings. The frozen zero-shot path has no gradient updates. Supervised comparison runs use AdamW, learning rate 1e-4, cosine annealing, 100 epochs, and early stopping. Storage depends on separately obtained datasets and model weights.

## Commands

```bash
python -m hicpg.cli inspect-prompts --config configs/main.yaml
python -m hicpg.cli split --manifest data/manifest.csv --seed 42
python -m hicpg.cli score --config configs/main.yaml --manifest data/manifest.csv --output outputs/scores.csv
python -m hicpg.cli calibrate --scores outputs/calibration.csv --alpha 0.10 --output outputs/calibrator.json
python -m hicpg.cli evaluate --scores outputs/evaluation.csv --calibrator outputs/calibrator.json --output outputs/metrics.json
python -m hicpg.cli compare --directory outputs/seeds --output outputs/summary.json
```

The twenty reference seeds are 42, 137, 256, 384, 512, 628, 743, 891, 1024, 1159, 1280, 1411, 1536, 1672, 1800, 1943, 2048, 2176, 2309, and 2456. Report 95% class-stratified bootstrap intervals using 10,000 resamples. Apply Bonferroni correction to the fifteen primary comparisons and Benjamini-Hochberg correction at q=0.05 to exploratory comparisons.

## Container

```bash
docker build -t hicpg:cuda118 .
docker run --gpus all --rm -v "$PWD:/workspace" hicpg:cuda118 python -m hicpg.cli inspect-prompts --config configs/main.yaml
```

The predictions are research outputs and are not a substitute for clinical diagnosis.
