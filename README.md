# 🫀 HeartWaveML - an automatic ECG Heartbeat Classification

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/📊-Kaggle_Notebook-blue.svg)](https://www.kaggle.com/code/josegarciamayen/heartwaveml)
[![Blog]( https://img.shields.io/badge/🔗-Code_explanation-red)](https://josegarciamayen.com/blog)

![HeartWaveML](https://github.com/JoseGarciaMayen/HeartWaveML/blob/main/assets/heartwaveml_title.png)

</div>

## Project Overview
This project implements an advanced machine learning pipeline for ***automated ECG heartbeat classification***, capable of detecting 5 types of cardiac conditions with clinical-grade accuracy. The system processes raw ECG signals and classifies heartbeats in <50 ms.

## Live Demo
👉 [Try the demo here](https://heartwaveml.josegarciamayen.com)

![API GIF](https://github.com/JoseGarciaMayen/HeartWaveML/blob/main/assets/heartwaveml.gif)

## Model results

> Metrics below use the **patient-wise inter-patient split** (DS1/DS2, see
> [Evaluation methodology](#evaluation-methodology)), no data leakage between
> train and test. Tree-based models (XGB/LGBM/ExtraTrees/CatBoost/CONVXGB)
> were dropped from this repo once the leakage was fixed and their honest
> scores plateaued at f1_macro ~0.51-0.58; see `MEJORAS.md` for that history.

<div align="center">

| Model | F1-N | F1-S | F1-V | F1-macro |
|-------|------|------|------|----------|
| CNN-MLP | 0.860 | 0.080 | 0.508 | 0.483 |
| BiLSTM (W=9) | 0.962 | 0.061 | 0.786 | 0.603 |
| Seq2Seq BiLSTM (W=45) | 0.969 | 0.333 | 0.848 | 0.717 |
| **Transformer (W=45)** | **0.975** | **0.647** | **0.869** | **0.830** |

</div>

The Transformer already exceeds the project target of f1_macro ≥ 0.80. See `CLAUDE.md` for full details.

## Evaluation methodology

The dataset is split **patient-wise** (the *inter-patient* paradigm): all
heartbeats from a given record (patient) are assigned to a single set, so no
patient appears in both train and test. This follows the de Chazal DS1/DS2
partition and is enforced in `split_data` (`src/data/splitter.py`) via the
fixed `DS1_RECORDS`/`DS2_RECORDS`/`CV_RECORDS` record sets (train/val come
from DS1, test is the fully unseen DS2).

This avoids **inter-patient data leakage**, where a random per-beat split lets
the model memorise patient-specific morphology and report optimistic metrics
that do not generalise to unseen patients. Expect lower (but honest) scores
than an intra-patient split.

## Features
- Data ***preprocessing*** and ***feature extraction*** from raw ECG signals.

- ***Tuning and training*** of various ML models using ***tensorflow***.

- Model ***evaluation*** using appropriate metrics for multiclass classification.

- Experiment tracking using ***ClearML***.

- ***Notebook*** for interactive experiments and visualization [here](https://www.kaggle.com/code/josegarciamayen/heartwaveml)

- ***DVC*** with ***Dagshub s3 bucket*** for data versioning and keeping track of our models.

- ***Docker + FastAPI*** to serve an easy-to-use interactive API.

- ***Continuous Integration*** (CI) using Github Actions.


## Quick Start
There are three ways to run ***HeartWaveML***:

### 1️⃣ Run only the API (via Docker)
If you only need the API, simply pull the [Docker image](https://hub.docker.com/r/josegm61/heartwaveml/tags) (<600MB):

```bash
docker pull josegm61/heartwaveml:latest
docker run -p 8000:8000 josegm61/heartwaveml:latest
```

The API will be running on http://localhost:8000
You can open `web/index.html` in your browser to interact with it. You can also see every endpoint at the [Swagger UI](http://localhost:8000/docs)

### 2️⃣ Use pretrained models and datasets (via DVC)
If you want to use the trained models and datasets:
```bash
dvc pull
pip install -r requirements.txt
```
This will fetch the models and datasets tracked with DVC and install dependencies (you probably will need a [Dagshub account](https://dagshub.com/))

### 3️⃣ Train models from scratch
If you prefer to generate the dataset and train the models yourself:
```bash
pip install -r requirements-dev.txt
# 1. generate datasets with the patient-wise split
python -m src.data.generate_data --mode deterministic
python -m src.data.generate_sequences
# 2. start ClearML, then tune + train
python -m src.pipeline tune transformer
python -m src.pipeline train transformer
```
Then serve the API with:
```bash
python -m src.api
```
This is the recommended option if you want to use this repo as a template to train your own models and try other combinations. See `CLAUDE.md` for the full model list (`cnn_mlp`, `lstm`, `seq2seq`, `transformer`) and pipeline commands.

## Model Design

<div align="center">

![Model Architecture](https://github.com/JoseGarciaMayen/HeartWaveML/blob/main/assets/heartwaveml.jpg)

</div>

## Project Structure
```
HeartWaveML/
├── .dvc/                         # DVC control files
├── .github/workflows/main.yml    # CI pipeline with GitHub Actions
├── assets/                       # Photos and videos
├── data/                         # Datasets (tracked in DVC)   
├── src/                          # Source code
│   ├── data/
│   │   ├── download_dataset.py   # Script to download dataset
│   │   └── generate_data.py      # Script to generate data
│   ├── saved_models/             # Trained models (tracked in DVC)   
│   ├── training/                 # Training logic
│   ├── tuning/                   # Hyperparameter tuning
│   ├── api.py                    # API to serve the model
│   ├── evaluate.py               # Model evaluation
│   ├── predict.py                # Run predictions on new data
│   ├── preprocessing.py          # Data preprocessing functions
│   └── utils.py                  # Helper functions
├── web/
│   └── index.html                # Web interface
├── .dockerignore                 # Ignore files in Docker builds
├── .gitignore                    # Ignore files in git
├── Dockerfile                    # Docker image definition
├── dvc.lock                      # Exact DVC state for data/pipelines
├── dvc.yaml                      # DVC pipeline definitions
├── LICENSE                       # Project license
├── README.md                     # Main documentation
├── requirements_api.txt          # API dependencies
└── requirements.txt              # Core dependencies


```
## Clinical Impact

This model provides a scalable solution for cardiac monitoring, combining ***clinical-grade*** reliability with unparalleled ***speed***.

- ***High-Accuracy Screening***: 98.5% accuracy ensures reliable detection of 5 types of cardiac conditions, a rate comparable to human experts.

- ***Real-Time Analysis***: With an average inference time of under 50 ms per heartbeat, the system enables real-time, continuous monitoring, and the rapid processing of massive datasets.

- ***Augments Professional Expertise***: By automating the initial screening process, the system frees up healthcare professionals to focus their expertise on complex cases and direct patient care.

## Dataset

We use the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), a widely used benchmark dataset for ECG signal classification. The dataset contains 48 half-hour recordings of two-lead ambulatory ECG signals sampled at 360 Hz. Each recording is annotated with beat labels, indicating the type of each heartbeat according to standard conventions.

Each ECG segment is resampled or cropped to 187 samples, then scaled and filtered. The process of filtering and scaling is a must to improve our models performance:

![Signal](https://github.com/JoseGarciaMayen/HeartWaveML/blob/main/assets/plot.png)

There are lots of heartbeats types:

![Type Distribution](https://github.com/JoseGarciaMayen/HeartWaveML/blob/main/assets/type_distribution.png)

So we map them into 5 classes:

```python
class_mapping = {
    'N': 0, '·': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,           # Normal beat
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                           # Supraventricular ectopic beat
    'V': 2, 'E': 2,                                           # Ventricular ectopic beat
    'F': 3,                                                   # Fusion beat
    '/': 4, 'f': 4, 'x': 4, 'Q': 4, '|': 4, '~': 4            # Unknown beat
}
```

And we end up with this distribution:

<div align="center">

| Class | Count |
|--------|-------|
| **0** | 90608 |
| **1** | 2781 |
| **2** | 7235 |
| **3** | 802 |
| **4** | 8981 |

</div>

We also applied SMOTE to fix the extreme class imbalance oversampling classes 1 and 3 to 5000 samples.

```python
sampling_strategy_dict = {
    3: 5000, 1: 5000
    }

    smote = SMOTE(sampling_strategy=sampling_strategy_dict, random_state=42, k_neighbors=5)
```

And then we split the data into train, validation and test. To do some tests, we created various datasets:

<div align="center">

| Dataset | Description |
|--------|-------|
| **base** | Scaled and filtered signal |
| **cnn** | Features extracted by CNN  |
| **feat** | Signal + Engineered features |
| **feat_only** | Engineered features |

</div>

## Citation

```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
```

```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
```

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements. For any questions, suggestions, or feedback, please don’t hesitate to contact me at josegarciamayen@gmail.com. Your advice and collaboration are greatly appreciated!