# 🎙️ DCMC-Fusion: Data-Centric and Model-Centric Ensemble Learning for Audio Deepfake Detection

**Dora María Ballesteros · Daniel Suárez**

---

## 📌 Description

**DCMC-Fusion** is an ensemble learning framework for audio deepfake detection that jointly explores two complementary perspectives: **data-centric fusion** and **model-centric fusion**.

The framework is built from **20 previously trained deep learning models**, obtained from the combination of five architectures and four audio representations:

- **Architectures:** DenseNet121, ResNet50, InceptionV3, ResNet34, and VGG16.
- **Audio representations:** LOG, MEL, DWT, and CQT.

Rather than directly combining the final decisions of these models, DCMC-Fusion uses their **inference probabilities as input features for a second-level classifier**.

The experimental design evaluates all possible combinations of architectures:

| K | Architectures per combination | Datasets | Features per dataset | Fusion configurations |
|---|---|---|---|---|
| 1 | 1 | 5 | 4 | 20 |
| 2 | 2 | 10 | 8 | 40 |
| 3 | 3 | 10 | 12 | 40 |
| 4 | 4 | 5 | 16 | 20 |
| 5 | 5 | 1 | 20 | 4 |
| **Total** | | **31** | | **124** |

For each dataset, four machine learning classifiers are evaluated as fusion mechanisms:

**Logistic Regression (LR), Random Forest (RF), XGBoost (XGB), and Support Vector Machine (SVM).**

This results in **124 experimental configurations**, enabling the analysis of three complementary aspects:

- **Data-centric analysis:** investigates the effects of architecture composition and the number of architectures combined.
- **Model-centric analysis:** examines the influence of the second-level fusion classifier.
- **Joint analysis:** evaluates the interaction between input feature configurations and fusion mechanisms.

In addition, the framework is compared with individual deep learning detectors and three probability/decision-level fusion baselines: **One-Hard Voting, Soft Voting, and Weighted Averaging**.

---

## 📂 Repository Structure

The repository is organized into three main directories: [`data/`](data/), [`notebooks/`](notebooks/), and [`results/`](results/).

```text
DCMC-Fusion/
│
├── data/
│   ├── best_hyperparameters_k1.json
│   ├── best_hyperparameters_k2.json
│   ├── best_hyperparameters_k3.json
│   ├── best_hyperparameters_k4.json
│   ├── best_hyperparameters_k5.json
│   ├── experiment_split.json
│   └── inference_probabilities.csv
│
├── notebooks/
│   ├── Best_Fusion_Configuration.ipynb
│   ├── Best_individual_model.ipynb
│   │
│   ├── DCMC_Fusion_K1_Timed.ipynb
│   ├── DCMC_Fusion_K2_Timed.ipynb
│   ├── DCMC_Fusion_K3_Timed.ipynb
│   ├── DCMC_Fusion_K4_Timed.ipynb
│   ├── DCMC_Fusion_K5_Timed.ipynb
│   │
│   ├── Fusion_Classifiers_Hyperparameter_Search_k1.ipynb
│   ├── Fusion_Classifiers_Hyperparameter_Search_k2.ipynb
│   ├── Fusion_Classifiers_Hyperparameter_Search_k3.ipynb
│   ├── Fusion_Classifiers_Hyperparameter_Search_k4.ipynb
│   ├── Fusion_Classifiers_Hyperparameter_Search_k5.ipynb
│   │
│   ├── One_Hard_Voting_Baseline.ipynb
│   ├── Soft_Voting_baseline.ipynb
│   ├── Weighted_voting_baseline.ipynb
│   │
│   ├── results_data_centric.ipynb
│   └── results_model_contric.ipynb
│
├── results/
│   ├── class_metrics_k1.csv
│   ├── class_metrics_k2.csv
│   ├── class_metrics_k3.csv
│   ├── class_metrics_k4.csv
│   ├── class_metrics_k5.csv
│   ├── class_metrics_one_hard.csv
│   │
│   ├── global_metrics_k1.csv
│   ├── global_metrics_k2.csv
│   ├── global_metrics_k3.csv
│   ├── global_metrics_k4.csv
│   ├── global_metrics_k5.csv
│   ├── global_metrics_one_hard.csv
│   │
│   ├── soft_voting_class_metrics.csv
│   ├── soft_voting_global_metrics.csv
│   ├── weighted_averaging_class_metrics.csv
│   └── weighted_averaging_global_metrics.csv
│
├── LICENSE
└── README.md
```

---

## 📁 `data/`

The [`data/`](data/) directory contains the input data and configuration files required to reproduce the experiments.

- [`inference_probabilities.csv`](data/inference_probabilities.csv) contains the inference probabilities obtained from the 20 previously trained deep learning models. These probabilities constitute the input features for DCMC-Fusion and the probability-based fusion baselines.

- [`experiment_split.json`](data/experiment_split.json) stores the predefined training and test partitions, including sample ordering and feature information. This ensures that experiments use a consistent data partition.

- **Hyperparameter configuration files:** [`best_hyperparameters_k1.json`](data/best_hyperparameters_k1.json), [`best_hyperparameters_k2.json`](data/best_hyperparameters_k2.json), [`best_hyperparameters_k3.json`](data/best_hyperparameters_k3.json), [`best_hyperparameters_k4.json`](data/best_hyperparameters_k4.json), and [`best_hyperparameters_k5.json`](data/best_hyperparameters_k5.json) contain the selected hyperparameters for the fusion classifiers at each value of K.

---

## 📓 `notebooks/`

The [`notebooks/`](notebooks/) directory contains the experimental implementation, baseline evaluations, model selection procedures, and results analysis.

### 1. Hyperparameter Selection

Hyperparameter selection is performed separately for each value of K.

- [`Fusion_Classifiers_Hyperparameter_Search_k1.ipynb`](notebooks/Fusion_Classifiers_Hyperparameter_Search_k1.ipynb) — hyperparameter selection for K=1.
- [`Fusion_Classifiers_Hyperparameter_Search_k2.ipynb`](notebooks/Fusion_Classifiers_Hyperparameter_Search_k2.ipynb) — hyperparameter selection for K=2.
- [`Fusion_Classifiers_Hyperparameter_Search_k3.ipynb`](notebooks/Fusion_Classifiers_Hyperparameter_Search_k3.ipynb) — hyperparameter selection for K=3.
- [`Fusion_Classifiers_Hyperparameter_Search_k4.ipynb`](notebooks/Fusion_Classifiers_Hyperparameter_Search_k4.ipynb) — hyperparameter selection for K=4.
- [`Fusion_Classifiers_Hyperparameter_Search_k5.ipynb`](notebooks/Fusion_Classifiers_Hyperparameter_Search_k5.ipynb) — hyperparameter selection for K=5.

The selected configurations are stored in the corresponding JSON files in the `data/` directory.

### 2. DCMC-Fusion Experiments

The DCMC-Fusion experiments are organized according to the number of architectures included in each dataset.

- [`DCMC_Fusion_K1_Timed.ipynb`](notebooks/DCMC_Fusion_K1_Timed.ipynb) — evaluates 5 datasets and 20 fusion configurations.
- [`DCMC_Fusion_K2_Timed.ipynb`](notebooks/DCMC_Fusion_K2_Timed.ipynb) — evaluates 10 datasets and 40 fusion configurations.
- [`DCMC_Fusion_K3_Timed.ipynb`](notebooks/DCMC_Fusion_K3_Timed.ipynb) — evaluates 10 datasets and 40 fusion configurations.
- [`DCMC_Fusion_K4_Timed.ipynb`](notebooks/DCMC_Fusion_K4_Timed.ipynb) — evaluates 5 datasets and 20 fusion configurations.
- [`DCMC_Fusion_K5_Timed.ipynb`](notebooks/DCMC_Fusion_K5_Timed.ipynb) — evaluates 1 dataset and 4 fusion configurations.

Together, these notebooks cover **31 datasets and 124 experimental configurations**.

### 3. Baseline Methods

Four baseline evaluations are considered to provide reference points for assessing the performance of DCMC-Fusion.

#### Individual Model Baseline

[`Best_individual_model.ipynb`](notebooks/Best_individual_model.ipynb)

Evaluates the 20 individual audio deepfake detectors independently on the fixed held-out test partition.

The models are ranked according to **F1-macro**, with Accuracy, Precision-macro, and Recall-macro also reported.

This establishes the performance of individual detectors without applying a fusion mechanism.

#### One-Hard Voting

[`One_Hard_Voting_Baseline.ipynb`](notebooks/One_Hard_Voting_Baseline.ipynb)

Evaluates a decision-level fusion baseline based on the hard classification decisions of individual detectors.

#### Soft Voting

[`Soft_Voting_baseline.ipynb`](notebooks/Soft_Voting_baseline.ipynb)

Evaluates a probability-based fusion baseline using the average of the synthetic-class scores produced by the selected individual models.

#### Weighted Averaging

[`Weighted_voting_baseline.ipynb`](notebooks/Weighted_voting_baseline.ipynb)

Evaluates a weighted probability aggregation approach in which the contributions of the four spectral representations are determined using rank-based weights.

The global representation weights are obtained by averaging the rank-based weights across the five architectures and normalizing them. These weights remain fixed across all values of K.

### 4. Best Fusion Configuration

[`Best_Fusion_Configuration.ipynb`](notebooks/Best_Fusion_Configuration.ipynb)

Analyzes the experimental results to identify the best-performing fusion configurations.

This notebook supports comparisons between fusion approaches and the identification of representative configurations for reporting.

### 5. Experimental Results Analysis

The experimental analysis is divided into two complementary perspectives.

#### Data-Centric Analysis

[`results_data_centric.ipynb`](notebooks/results_data_centric.ipynb)

Examines the influence of input dataset construction on classification performance, considering:

- **Composition effect:** differences among architecture combinations with the same K.
- **Size effect:** differences associated with the number of architectures combined.

#### Model-Centric Analysis

[`results_model_contric.ipynb`](notebooks/results_model_contric.ipynb)

Examines the performance of the four fusion classifiers, supporting the comparison of different second-level learning mechanisms.

---

## 📊 `results/`

The [`results/`](results/) directory contains the experimental metrics for DCMC-Fusion and the voting-based baselines.

Results are provided at two levels:

- **Global metrics:** Accuracy, Precision-macro, Recall-macro, and F1-macro.
- **Class-specific metrics:** classification performance reported separately for natural and synthetic audio.

These files allow the experimental results to be inspected without rerunning the corresponding notebooks.

### 1. DCMC-Fusion Global Metrics

- [`global_metrics_k1.csv`](results/global_metrics_k1.csv)
- [`global_metrics_k2.csv`](results/global_metrics_k2.csv)
- [`global_metrics_k3.csv`](results/global_metrics_k3.csv)
- [`global_metrics_k4.csv`](results/global_metrics_k4.csv)
- [`global_metrics_k5.csv`](results/global_metrics_k5.csv)

### 2. DCMC-Fusion Class-Specific Metrics

- [`class_metrics_k1.csv`](results/class_metrics_k1.csv)
- [`class_metrics_k2.csv`](results/class_metrics_k2.csv)
- [`class_metrics_k3.csv`](results/class_metrics_k3.csv)
- [`class_metrics_k4.csv`](results/class_metrics_k4.csv)
- [`class_metrics_k5.csv`](results/class_metrics_k5.csv)

### 3. Voting Baseline Results

| Method | Global metrics | Class-specific metrics |
|---|---|---|
| One-Hard Voting | [`global_metrics_one_hard.csv`](results/global_metrics_one_hard.csv) | [`class_metrics_one_hard.csv`](results/class_metrics_one_hard.csv) |
| Soft Voting | [`soft_voting_global_metrics.csv`](results/soft_voting_global_metrics.csv) | [`soft_voting_class_metrics.csv`](results/soft_voting_class_metrics.csv) |
| Weighted Averaging | [`weighted_averaging_global_metrics.csv`](results/weighted_averaging_global_metrics.csv) | [`weighted_averaging_class_metrics.csv`](results/weighted_averaging_class_metrics.csv) |

---

## 🔬 Experimental Evaluation

The experimental framework uses a fixed training/test partition to ensure consistency across evaluations.

The main classification metrics are:

- **Accuracy:** overall proportion of correctly classified samples.
- **Precision-macro:** unweighted average of class-specific precision values.
- **Recall-macro:** unweighted average of class-specific recall values.
- **F1-macro:** unweighted average of class-specific F1-scores.

Macro-averaged metrics are used to evaluate classification performance across both natural and synthetic audio classes without assigning greater importance to either class.

The evaluation includes:

1. Performance of the 20 individual deep learning detectors.
2. Performance of the 124 DCMC-Fusion configurations.
3. Comparison with One-Hard Voting, Soft Voting, and Weighted Averaging.
4. Data-centric analysis of architecture composition and combination size.
5. Model-centric analysis of the four fusion classifiers.

Together, these experiments provide a systematic assessment of the effects of input feature construction and fusion mechanism selection on audio deepfake detection performance.

