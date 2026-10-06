# DCMC-Fusion: Data-Centric and Model-Centric Ensemble Learning for Audio Deepfake Detection

**Dora María Ballesteros · Daniel Suárez · **

---

## 📌 Description

**DCMC-Fusion** is an ensemble learning framework for audio deepfake detection that jointly explores two complementary perspectives: **data-centric fusion** and **model-centric fusion**.

The framework is built from **20 pretrained deep learning models**, obtained from the combination of five architectures and four audio representations:

- **Architectures:** DenseNet121, ResNet50, InceptionV3, ResNet34, and VGG16.
- **Audio representations:** LOG, MEL, DWT, and CQT.

Rather than directly combining the final decisions of these models, DCMC-Fusion uses their **inference probabilities as input features for a second-level classifier**.

The experimental design evaluates combinations of:

- **1 architecture (K=1)**
- **2 architectures (K=2)**
- **3 architectures (K=3)**
- **4 architectures (K=4)**
- **5 architectures (K=5)**

For each combination, four machine learning classifiers are evaluated as fusion mechanisms:

**Logistic Regression (LR), Random Forest (RF), XGBoost (XGB), and Support Vector Machine (SVM).**

This results in **124 experimental configurations**, enabling the joint analysis of the effect of both the **composition of the input data** and the **fusion classifier**.

---

## 📂 Repository Structure

The repository is organized into three main directories: [`data/`](data/), [`notebooks/`](notebooks/), and [`results/`](results/).

```text
DCMC-Fusion/
│
├── data/
│   ├── best_hyperparameters.json
│   ├── experiment_split.json
│   └── inference_probabilities.csv
│
├── notebooks/
│   ├── Fusion_Classifiers_Hyperparameter_Search.ipynb
│   ├── DCMC_Fusion_K1.ipynb
│   ├── DCMC_Fusion_K2.ipynb
│   ├── DCMC_Fusion_K3.ipynb
│   ├── DCMC_Fusion_K4.ipynb
│   └── DCMC_Fusion_K5.ipynb
│
├── results/
│   ├── global_metrics_k1.csv
│   ├── global_metrics_k2.csv
│   ├── global_metrics_k3.csv
│   ├── global_metrics_k4.csv
│   └── global_metrics_k5.csv
│
├── LICENSE
└── README.md

### 📁 `data/`

The [`data/`](data/) directory contains the input data and configuration files required to reproduce the experiments.

- [`inference_probabilities.csv`](data/inference_probabilities.csv) contains the inference probabilities obtained from the pretrained deep learning models. These probabilities constitute the input features of the DCMC-Fusion classifiers.
- [`best_hyperparameters.json`](data/best_hyperparameters.json) contains the selected hyperparameters for the four fusion classifiers.
- [`experiment_split.json`](data/experiment_split.json) stores the train/test partition used throughout the experiments, ensuring that all configurations are evaluated using the same data split.

### 📓 `notebooks/`

The [`notebooks/`](notebooks/) directory contains the complete experimental implementation.

The notebook [`Fusion_Classifiers_Hyperparameter_Search.ipynb`](notebooks/Fusion_Classifiers_Hyperparameter_Search.ipynb) performs the hyperparameter selection for the fusion classifiers.

The remaining notebooks implement the DCMC-Fusion experiments according to the number of architectures included in each configuration:

- [`DCMC_Fusion_K1.ipynb`](notebooks/DCMC_Fusion_K1.ipynb) — combinations with **1 architecture**.
- [`DCMC_Fusion_K2.ipynb`](notebooks/DCMC_Fusion_K2.ipynb) — combinations with **2 architectures**.
- [`DCMC_Fusion_K3.ipynb`](notebooks/DCMC_Fusion_K3.ipynb) — combinations with **3 architectures**.
- [`DCMC_Fusion_K4.ipynb`](notebooks/DCMC_Fusion_K4.ipynb) — combinations with **4 architectures**.
- [`DCMC_Fusion_K5.ipynb`](notebooks/DCMC_Fusion_K5.ipynb) — configuration with **all 5 architectures**.

### 📊 `results/`

The [`results/`](results/) directory provides the **global metrics obtained for each value of K**:

- [`global_metrics_k1.csv`](results/global_metrics_k1.csv)
- [`global_metrics_k2.csv`](results/global_metrics_k2.csv)
- [`global_metrics_k3.csv`](results/global_metrics_k3.csv)
- [`global_metrics_k4.csv`](results/global_metrics_k4.csv)
- [`global_metrics_k5.csv`](results/global_metrics_k5.csv)

These files are included so that the main experimental results can be inspected directly **without rerunning the notebooks**.

When the experimental notebooks are executed, additional detailed result files are automatically generated.
