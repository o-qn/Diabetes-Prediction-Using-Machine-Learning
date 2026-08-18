[README.md — Diabetes Prediction Analysis.md](https://github.com/user-attachments/files/31196271/README.md.Diabetes.Prediction.Analysis.md)
# 🩺 Diabetes Prediction Using Machine Learning

A machine learning classification project that predicts whether a patient has diabetes using demographic and clinical information.

The project covers the complete machine learning workflow, including **data cleaning, exploratory data analysis, preprocessing, class imbalance handling, model training, hyperparameter tuning, evaluation, and final model selection**.

---

## 📌 Project Overview

Diabetes is a chronic disease that can lead to serious health complications if it remains undiagnosed. Early identification of individuals at risk can support timely medical intervention and lifestyle changes.

The objective of this project is to build a machine learning model that predicts whether a patient has diabetes based on several demographic and clinical features.

### Target

- `0` — No Diabetes
- `1` — Diabetes

---

## 📊 Dataset

The dataset contains **100,000 patient records** and the following features:

| Feature | Description |
|---|---|
| `gender` | Patient gender |
| `age` | Patient age |
| `hypertension` | Whether the patient has hypertension |
| `heart_disease` | Whether the patient has heart disease |
| `smoking_history` | Patient smoking history |
| `bmi` | Body Mass Index |
| `HbA1c_level` | HbA1c blood test level |
| `blood_glucose_level` | Blood glucose level |
| `diabetes` | Target variable |

### Data Cleaning

Initial inspection showed:

- **100,000** original rows
- **0 missing values**
- **3,854 duplicate rows**
- **96,146 rows** remaining after duplicate removal

---

## ⚖️ Class Imbalance

The target variable is strongly imbalanced:

- **91.18%** — No Diabetes
- **8.82%** — Diabetes

Because of this imbalance, accuracy alone is not a reliable evaluation metric. A model that predicts "No Diabetes" for every patient could achieve approximately 91% accuracy while failing to identify diabetic patients.

For this reason, the project focuses on:

- Precision
- Recall
- F1-Score
- ROC-AUC

To address the imbalance during training, the project uses:

- **SMOTE** for minority-class oversampling
- **RandomUnderSampler** for majority-class undersampling

Both techniques are applied inside the training pipeline to prevent data leakage.

---

## 🔍 Exploratory Data Analysis

Exploratory analysis was performed to understand the distribution of the target and relationships between patient characteristics and diabetes.

### Strongest Correlations with Diabetes

| Feature | Correlation |
|---|---:|
| Blood Glucose Level | 0.42 |
| HbA1c Level | 0.41 |
| Age | 0.26 |
| BMI | 0.21 |
| Hypertension | 0.20 |
| Heart Disease | 0.17 |
| Smoking History | 0.09 |
| Gender | 0.04 |

### Key EDA Findings

- **HbA1c level** and **blood glucose level** provide the clearest separation between diabetic and non-diabetic patients.
- Diabetic patients tend to be **older**.
- Patients with diabetes generally have a somewhat **higher BMI**.
- Patients with **hypertension** or **heart disease** show a higher proportion of diabetes.
- Gender and smoking history have relatively weak linear correlations with the target.

---

## 🛠️ Data Preprocessing

A leak-free machine learning pipeline was created using `ColumnTransformer` and `imblearn.pipeline.Pipeline`.

### Numerical Features

Numerical variables are transformed using:

```python
StandardScaler()
```

This includes:

- Age
- Hypertension
- Heart disease
- BMI
- HbA1c level
- Blood glucose level

### Categorical Features

Categorical variables are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This includes:

- Gender
- Smoking history

### Train/Test Split

The cleaned dataset is divided into:

- **80% training data**
- **20% testing data**

with:

```python
stratify=y
```

to preserve the original diabetes class distribution.

Importantly, the train/test split occurs **before resampling** to avoid data leakage.

---

## 🤖 Machine Learning Models

Four classification algorithms were initially evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)

Each model uses the same general pipeline:

```text
Preprocessing
     ↓
SMOTE
     ↓
Random Under-Sampling
     ↓
Classifier
```

---

## 📈 Baseline Model Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.943 | 0.642 | 0.800 | **0.712** |
| Decision Tree | 0.929 | 0.572 | 0.788 | 0.663 |
| KNN | 0.892 | 0.442 | 0.851 | 0.582 |
| Logistic Regression | 0.884 | 0.424 | **0.881** | 0.573 |

Random Forest achieved the highest baseline F1-score.

Logistic Regression achieved the highest recall but generated significantly more false-positive predictions, resulting in lower precision.

---

## 🔧 Hyperparameter Tuning

The strongest models were further optimized using cross-validation.

### Random Forest

`GridSearchCV` was used with:

```python
n_estimators = [100, 200]
max_depth = [10, 20, None]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
```

Best parameters:

```python
{
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "n_estimators": 200
}
```

### Decision Tree

`GridSearchCV` explored:

```python
criterion = ["gini", "entropy"]
max_depth = [5, 10, 15, None]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
```

### KNN

`RandomizedSearchCV` was used to tune:

```python
n_neighbors
weights
p
```

The best KNN configuration used:

```python
n_neighbors = 4
weights = "distance"
p = 2
```

---

## 🏆 Final Model

### Tuned Random Forest

The **Tuned Random Forest** was selected as the final model because it achieved the strongest overall balance between precision, recall, F1-score, and ROC-AUC.

| Metric | Score |
|---|---:|
| Train Accuracy | 0.9765 |
| Test Accuracy | **0.9432** |
| Precision | **0.6414** |
| Recall | **0.8078** |
| F1-Score | **0.7150** |
| ROC-AUC | **0.9698** |

### Why Random Forest?

For an imbalanced medical screening problem, correctly identifying positive cases is especially important.

The tuned Random Forest provides:

- Strong recall for detecting diabetic patients
- Better precision than the high-recall Logistic Regression and KNN models
- Highest overall F1-score
- Excellent ROC-AUC
- Ability to capture nonlinear relationships between clinical measurements

---

## 🔬 Feature Importance

The final Random Forest model identified the following features as the most influential:

| Feature | Importance |
|---|---:|
| HbA1c Level | **0.367** |
| Blood Glucose Level | **0.278** |
| Age | **0.181** |
| BMI | **0.108** |
| Hypertension | 0.024 |
| Heart Disease | 0.014 |

The results confirm the findings from the exploratory data analysis: **HbA1c level and blood glucose level dominate the model's predictions**.

---

## 💻 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- SciPy
- Jupyter Notebook

---

## 📁 Project Structure

```text
diabetes-prediction/
│
├── diabetes_prediction_analysis.ipynb
├── diabetes_prediction_dataset.csv
├── Diabetes_Prediction_Analysis(1).pptx
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd diabetes-prediction
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scipy jupyter
```

### 3. Make sure the dataset is available

Place:

```text
diabetes_prediction_dataset.csv
```

in the same directory as the notebook.

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
diabetes_prediction_analysis.ipynb
```

and run the cells in order.

---

## 📌 Key Takeaways

- The diabetes target is highly imbalanced, so accuracy alone is misleading.
- SMOTE and random under-sampling were incorporated inside the model pipeline to handle imbalance without introducing data leakage.
- HbA1c level and blood glucose level are the strongest predictors.
- Random Forest performed better overall than Logistic Regression, Decision Tree, and KNN.
- Hyperparameter tuning further improved Random Forest performance.
- The final tuned model achieved an **F1-score of 0.715** and **ROC-AUC of 0.970**.

---

## ⚠️ Disclaimer

This project was developed for **educational and machine learning purposes**. The model should not be used as a substitute for professional medical diagnosis, clinical testing, or medical advice.

---

## 📄 Project Files

The repository includes:

- Full Jupyter Notebook containing EDA, preprocessing, modeling, and evaluation
- Project presentation summarizing the analysis and results
- Dataset used for model development

---

## 👤 Author

**Data Science / Machine Learning Project**

Feel free to explore the notebook, reproduce the analysis, and experiment with additional models or optimization techniques.

---

⭐ If you found this project useful, consider giving the repository a star!
