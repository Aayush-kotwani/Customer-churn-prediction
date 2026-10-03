# 🏦 Customer Churn Prediction

### End-to-End Machine Learning | Custom Imputation | Feature Engineering | Model Optimization

An end-to-end machine learning project for predicting customer churn using **data-driven preprocessing, custom imputation, feature engineering, ensemble modelling, and hyperparameter optimization**.

The project goes beyond a basic classification workflow by benchmarking multiple algorithms and tuning **LightGBM, XGBoost, SVC, and CatBoost**.

---

## 🚀 Highlights

* 📊 90,000 customer records
* 🔍 Exploratory Data Analysis & data-quality analysis
* 🧹 Custom distribution-aware imputation
* ⭐ Domain-driven feature engineering
* ⚖️ Class-imbalance handling
* ⚙️ `Pipeline` + `ColumnTransformer`
* 🤖 9 baseline models
* 🎯 F1-focused evaluation
* 🔎 5-fold `RandomizedSearchCV`
* ⚡ LightGBM & XGBoost optimization
* 🐈 CatBoost with native categorical features
* 🏁 Final model retrained on the complete training dataset

---

## 📌 Problem Statement

Predict whether a customer will **churn/exit** based on demographic, financial, account, and engagement-related information.

**Target:** `exit_status`

Because the target is imbalanced, **F1-score** is used as the primary model-selection metric.

---

## 📊 Dataset

| Property         |                 Value |
| ---------------- | --------------------: |
| Training Samples |                90,000 |
| Features         |   14 original columns |
| Problem          | Binary Classification |
| Target           |         `exit_status` |

Features include credit score, country, gender, age, tenure, account balance, product count, card ownership, activity, and salary.

---

# 🔄 Workflow

```text
Raw Data
   ↓
EDA & Data Quality
   ↓
Custom Imputation
   ↓
Feature Engineering
   ↓
Stratified Train/Validation Split
   ↓
Preprocessing Pipeline
   ↓
Baseline Benchmarking
   ↓
Hyperparameter Optimization
   ↓
LightGBM / XGBoost / CatBoost
   ↓
Full-Data Retraining
   ↓
Final Predictions
```

---

# 🧹 Custom Data Preprocessing

A major focus of this project is **feature-specific imputation** rather than applying one generic strategy.

### Country — Proportional Imputation

Missing `country` values are filled by randomly sampling according to the existing category distribution rather than assigning every missing value to the mode.

Resulting distribution:

```text
France     57.12%
Spain      21.98%
Germany    20.91%
```

### Account Balance — Zero-Inflated Imputation

`acc_balance` contains a significant number of zero values. A custom strategy preserves the observed probability of zero while sampling non-zero values from the observed distribution.

This helps maintain the underlying structure of the feature.

---

# ⭐ Feature Engineering

The project creates several domain-driven features:

| Feature                  | Purpose                                           |
| ------------------------ | ------------------------------------------------- |
| `zero_balance_flag`      | Identifies zero-balance customers                 |
| `is_senior_risk`         | Captures an age-based segment                     |
| `engagement_score`       | Combines card ownership and activity              |
| `inactive_with_products` | Detects inactive customers with multiple products |
| `germany_female`         | Country × gender interaction                      |
| `tenure_per_age`         | Tenure relative to age                            |
| `products_per_tenure`    | Products relative to tenure                       |

These features attempt to capture behavioural relationships that may not be directly represented by the raw variables.

---

# 🤖 Baseline Model Performance

The initial benchmark produced:

| Model                       |            F1 |       ROC-AUC |      Accuracy |
| --------------------------- | ------------: | ------------: | ------------: |
| **LightGBM**                |    **0.6311** |        0.8739 |        0.8197 |
| XGBoost                     |        0.6250 |        0.8668 |    **0.8214** |
| SVC                         |        0.6109 |        0.8476 |        0.7937 |
| Gradient Boosting           |        0.5979 |    **0.8743** |        0.8567 |
| Decision Tree               |        0.5059 |        0.6866 |        0.7911 |
| Random/Extra Trees & others | 0.5547–0.5832 | 0.8011–0.8531 | 0.7555–0.8502 |

**Key observation:** LightGBM achieved the highest baseline F1 of **0.6311**.

---

# 🎯 Hyperparameter Optimization

The strongest models were further optimized using:

```text
RandomizedSearchCV
        +
5-Fold Stratified Cross-Validation
        +
F1 Scoring
```

### Tuned Results

| Model           | Best Validation F1 |
| --------------- | -----------------: |
| 🥇 **LightGBM** |         **0.6524** |
| XGBoost         |             0.6522 |
| CatBoost        |             0.6506 |
| SVC             |             0.6190 |

LightGBM improved from **0.6311 → 0.6524 F1** after tuning.

---

# 🐈 Going Beyond XGBoost & LightGBM — CatBoost

The project also experiments with **CatBoost**, using native categorical handling for:

```text
gender
country
```

Instead of one-hot encoding these variables, CatBoost receives them as categorical features directly.

Best configuration:

```text
iterations: 200
depth: 4
learning_rate: 0.1
l2_leaf_reg: 1
scale_pos_weight: 2
```

Best validation F1:

```text
0.6506
```

This provides an additional comparison against the tuned LightGBM and XGBoost models.

---

# 🏆 Final Model Comparison

After fitting the tuned estimators on the validation workflow:

| Model        | Validation F1 |
| ------------ | ------------: |
| **LightGBM** |    **0.6381** |
| CatBoost     |        0.6373 |
| XGBoost      |        0.6371 |

The project proceeds with the tuned **LightGBM** model and retrains it on **100% of the labelled training data** before generating the final test predictions.

---

# 💡 Why This Project Stands Out

### Custom Imputation

Uses distribution-aware strategies instead of blindly applying mean/mode imputation.

### Feature Engineering

Creates behavioural, interaction, and ratio-based features from the original customer attributes.

### Extensive Benchmarking

Nine different classification algorithms are evaluated before optimization.

### Advanced Boosting

LightGBM, XGBoost, and CatBoost are all tuned and compared.

### Native Categorical Modelling

CatBoost is given a dedicated preprocessing pipeline to exploit its native categorical feature handling.

### Robust Validation

Uses stratified splitting and 5-fold cross-validation with F1 as the optimization metric.

---

# 🛠️ Tech Stack

**Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · LightGBM · XGBoost · CatBoost**

**Techniques:**
EDA · Custom Imputation · Feature Engineering · Pipelines · Class Imbalance · Cross-Validation · RandomizedSearchCV · Ensemble Learning

---

# 📁 Repository Structure

```text
Customer-churn-prediction/
│
├── customer_churn_prediction.ipynb
└── README.md
```

---

# ▶️ Run the Project

```bash
git clone https://github.com/Aayush-kotwani/Customer-churn-prediction.git
cd Customer-churn-prediction
jupyter notebook customer_churn_prediction.ipynb
```

The notebook was developed using a Kaggle dataset/environment, so local execution requires updating the dataset paths.

---

## 👨‍💻 Author

**Aayush Kotwani**

Data Scientist

**Interests:** Machine Learning · Data Science · Deep Learning · Generative AI · Software Development

---

> **The goal of this project was not simply to train a model, but to build a thoughtful ML pipeline where data quality, feature engineering, preprocessing, model selection, and optimization work together to improve churn prediction.**
