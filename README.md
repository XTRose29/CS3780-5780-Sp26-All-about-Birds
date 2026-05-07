# CS3780/5780 Sp26: All about Birds

Data and starter workflows for the CS3780/5780 Spring 2026 "All about Birds" Kaggle competition https://www.kaggle.com/competitions/cs-3780-5780-spring-2026-all-about-birds/leaderboard?tab=private. The task is to classify bird audio examples into broad bird groups using metadata, audio preprocessing, spectrogram features, and model training notebooks.

## Repository Contents

```text
.
|-- data/
|   |-- meta_info/                 # Source metadata and label mapping files
|   |-- reduced_64x32_new/         # Kaggle-ready reduced spectrogram dataset
|   |-- *.py                       # Data preparation, feature export, and baseline scripts
|   `-- pca_coeff_logreg_metrics.json
|-- train/
|   |-- train_ast.ipynb            # Audio Spectrogram Transformer workflow
|   |-- train_cnn.ipynb            # CNN workflow
|   |-- train_logreg.ipynb         # Logistic regression workflow
|   |-- train_vit*.ipynb           # Vision Transformer workflows
|   `-- fine_tune.ipynb
`-- README.md
```

## Included Dataset

The included Kaggle-style image dataset is at:

```text
data/reduced_64x32_new/All_about_Birds/
|-- train/<label>/*.png
`-- test/*.png
```

It contains 21,829 labeled training PNGs and 5,454 test PNGs. Labels in the training split are:

- `Antbird`
- `Bird of prey`
- `Flycatcher`
- `Ground bird`
- `Nocturnal bird`
- `Other non-passerine bird`
- `Parrot`
- `Tanager`
- `Water-associated bird`

The submission files are:

- `data/reduced_64x32_new/sample_submission.csv`: blank labels for Kaggle submission format.
- `data/reduced_64x32_new/solution.csv`: labels and public/private split for evaluation setup.

Metadata is stored in `data/meta_info/`, including `train.csv`, `train_with_class.csv`, `taxonomy.csv`, `recording_location.txt`, and `raw_spectrogram_sizes.csv`.

## Environment

Python 3.10+ is recommended. The preprocessing scripts also require `ffmpeg` and `ffprobe` on your `PATH`.

Install the Python packages used by the scripts and notebooks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy pillow scikit-learn matplotlib tqdm jupyter
```

Install `ffmpeg` separately if needed:

```bash
brew install ffmpeg
```

## Working With the Included Spectrogram Dataset

Open the notebooks in `train/` and point them at:

```text
data/reduced_64x32_new/All_about_Birds
```

Start Jupyter from the repo root:

```bash
jupyter notebook train/
```

## Regenerating Data From Raw Audio

Several scripts expect raw OGG audio in a local folder grouped by metadata filename. If you have the raw competition audio, a typical repo-root workflow is:

```bash
python data/build_new_train_audio.py \
  --csv-path data/meta_info/train_with_class.csv \
  --source-root data/train_audio \
  --output-root data/new_train_audio

python data/prepare_20s_data.py \
  --data-root data/new_train_audio \
  --output-root data/new_train_audio_20s
```

Export reduced 64x32 spectrogram images:

```bash
python data/export_lowdim.py \
  --data-root data/new_train_audio \
  --output-root data/lowdim_export
```

Run the low-dimensional logistic regression baseline on the exported variants before creating a Kaggle split:

```bash
python data/train_logreg_lowdim_export.py \
  --data-root data/lowdim_export \
  --output data/lowdim_export_logreg_metrics.json
```

Create a train/test split and Kaggle-style CSV files:

```bash
python data/split_lowdim_export.py \
  --data-root data/lowdim_export/reduced_64x32 \
  --label-source-root data/new_train_audio
```

## Other Feature Export Scripts

The `data/` directory includes alternative exports and baselines:

- `export_raw_spectrograms.py`: raw STFT spectrogram images.
- `export_new_train_audio_spectrograms.py`: spectrogram previews with configurable colormap.
- `export_ab_spectrograms.py`: A/B spectrogram variants for comparing preprocessing choices.
- `export_mfcc_13.py`: 13-coefficient MFCC image export.
- `export_pca_spectrograms.py`: PCA-compressed spectrogram export.
- `train_logreg.py`: logistic regression on handcrafted spectrogram and cepstral summary features.
- `train_logreg_pca_coeffs.py`: logistic regression on PCA coefficients.

Most scripts have defaults designed for local experimentation. Prefer passing explicit `--data-root`, `--output-root`, and related paths from the repo root so outputs are written where you expect.

## Notes

- Generated datasets and model outputs can be large. Keep raw audio, generated spectrogram folders, model pickles, and notebook checkpoints out of commits unless they are intentionally part of the release.
- `data/pca_coeff_logreg_metrics.json` records one baseline PCA/logistic-regression run using 100 files per label and 64 PCA components.
