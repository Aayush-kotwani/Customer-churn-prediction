# 🏦 Customer Churn Prediction

### End-to-End Machine Learning | Feature Engineering | Custom Imputation | Model Optimization

An end-to-end machine learning project for predicting customer churn using **data-driven preprocessing, custom imputation, feature engineering, ensemble modelling, and hyperparameter optimization**.

The project goes beyond a basic classification workflow by experimenting with multiple algorithms and specifically tuning **LightGBM, XGBoost, and CatBoost**.

---

## 🚀 Highlights

* 📊 90,000 customer records
* 🔍 Comprehensive EDA and data-quality analysis
* 🧹 Custom, distribution-aware imputation
* ⭐ Domain-driven feature engineering
* ⚖️ Class-imbalance handling
* 🔄 Stratified train-validation split
* ⚙️ Scikit-learn `Pipeline` + `ColumnTransformer`
* 🤖 9 baseline classification models
* 🎯 F1-focused model evaluation
* 🔎 5-fold `RandomizedSearchCV`
* ⚡ LightGBM & XGBoost optimization
* 🐈 CatBoost with native categorical features
* 🏁 Final model retrained on 100% of the training data

---

# 📌 Problem Statement

The objective is to predict whether a customer will **exit/churn** based on demographic, financial, account, and engagement-related information.

**Target:** `exit_status`

This is a binary classification problem with an imbalanced target, making **F1-score** particularly important during model selection.

---

# 📊 Dataset

The training dataset contains:

| Property |                 Value |
| -------- | --------------------: |
| Rows     |                90,000 |
| Columns  |                    14 |
| Problem  | Binary Classification |
| Target   |         `exit_status` |

Features include:

`credit_score`, `country`, `gender`, `age`, `tenure`, `acc_balance`, `prod_count`, `has_card`, and `is_active`.

---

# 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Understanding & EDA
   ↓
Data Quality Checks
   ↓
Custom Imputation
   ↓
Feature Engineering
   ↓
Stratified Train/Validation Split
   ↓
Preprocessing Pipeline
   ↓
Baseline Model Benchmarking
   ↓
Hyperparameter Optimization
   ↓
LightGBM vs XGBoost vs CatBoost
   ↓
Full-Data Retraining
   ↓
Final Predictions
```

---

# 🧹 Data Cleaning & Custom Imputation

Instead of applying a single generic imputation strategy, different techniques are used according to the behaviour of each feature.

### `credit_score`

Missing values → **Mean Imputation**

### `prod_count`

Missing values → **Most-Frequent Imputation**

### `country`

A custom **proportional random imputation** method is implemented.

Instead of replacing every missing country with the mode, missing values are sampled according to the observed category distribution.

```text
Observed Distribution
        ↓
Category Probabilities
        ↓
Random Sampling
        ↓
Imputed Country
```

### `acc_balance`

Account balance is **zero-inflated**, so a custom strategy is used:

* Estimate the probability of zero balance.
* Preserve the observed non-zero distribution.
* Generate missing values using these learned characteristics.

This avoids blindly replacing missing balances with the mean/median.

---

# ⭐ Feature Engineering

Several domain-driven features are created to capture relationships that may not be obvious from the raw columns.

| Feature                  | Purpose                                                 |
| ------------------------ | ------------------------------------------------------- |
| `zero_balance_flag`      | Identifies customers with zero account balance          |
| `is_senior_risk`         | Captures a specific age-based risk segment              |
| `engagement_score`       | Combines card ownership and activity                    |
| `inactive_with_products` | Identifies inactive customers holding multiple products |
| `germany_female`         | Captures country × gender interaction                   |
| `tenure_per_age`         | Normalizes tenure relative to age                       |
| `products_per_tenure`    | Measures product ownership relative to tenure           |

The feature engineering logic is encapsulated inside a reusable function and applied consistently to both training and test data.

---

# ⚙️ Preprocessing

The project uses:

* `Pipeline`
* `ColumnTransformer`
* `SimpleImputer`
* `OneHotEncoder`
* `StandardScaler`

Categorical and numerical features receive separate preprocessing before being passed to the models.

The preprocessing pipeline is fitted only on training data and then applied to validation/test data to avoid inconsistent transformations.

---

# 🤖 Model Benchmarking

The following **9 models** are evaluated:

1. Logistic Regression
2. KNN
3. SVC
4. Decision Tree
5. Random Forest
6. Extra Trees
7. Gradient Boosting
8. LightGBM
9. XGBoost

Each model is evaluated using:

* **F1 Score**
* **ROC-AUC**
* **Accuracy**

F1 is used as the primary comparison metric because of the class imbalance.

---

# 🏆 Model Performance

### Baseline Models

> **Note:** The current GitHub notebook does not contain saved output values for this table. The code calculates these metrics, but the output cells are empty.

| Model               | F1 | ROC-AUC | Accuracy |
| ------------------- | -: | ------: | -------: |
| Logistic Regression |  — |       — |        — |
| KNN                 |  — |       — |        — |
| SVC                 |  — |       — |        — |
| Decision Tree       |  — |       — |        — |
| Random Forest       |  — |       — |        — |
| Extra Trees         |  — |       — |        — |
| Gradient Boosting   |  — |       — |        — |
| LightGBM            |  — |       — |        — |
| XGBoost             |  — |       — |        — |

**Primary metric:** F1 Score.

---

# 🎯 Hyperparameter Optimization

The strongest boosting approaches are further optimized using:

```text
RandomizedSearchCV
        +
5-Fold Stratified Cross-Validation
        +
F1 Scoring
```

### LightGBM

40 random configurations are tested across parameters including:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `num_leaves`
* `scale_pos_weight`
* `subsample`
* `colsample_bytree`
* `reg_alpha`
* `reg_lambda`

### XGBoost

40 configurations are explored across:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `min_child_weight`
* `subsample`
* `colsample_bytree`
* `scale_pos_weight`
* Regularization parameters

### CatBoost 🐈

The project goes one step further by training **CatBoost with native categorical features** rather than forcing `gender` and `country` through one-hot encoding.

25 configurations are searched across:

* `iterations`
* `depth`
* `learning_rate`
* `l2_leaf_reg`
* `scale_pos_weight`

All three tuned models are then compared on the same validation set using F1.

---

# 🥇 Tuned Model Comparison

| Model    | Validation F1 |
| -------- | ------------: |
| LightGBM |         **—** |
| XGBoost  |         **—** |
| CatBoost |         **—** |

The notebook compares the three tuned models directly before final model selection.

> **Final submission model:** Tuned **LightGBM**, retrained on the complete training dataset before generating `submission.csv`.

---

# 💡 Why This Project Stands Out

### 1. Custom Imputation

Missing values are handled according to the statistical behaviour of each feature instead of using one generic method.

### 2. Feature Engineering

Raw customer attributes are transformed into behavioural, interaction, and ratio-based features.

### 3. Multiple Models

The project benchmarks nine different classification approaches before optimization.

### 4. Going Beyond XGBoost & LightGBM

**CatBoost** is additionally trained with native categorical feature support.

### 5. Proper Validation

Stratified splitting and 5-fold cross-validation are used to make model comparison more robust.

### 6. Full-Data Retraining

The final LightGBM model is retrained using **100% of the available labelled training data** before producing the final submission.

---

# 🛠️ Tech Stack

**Languages & Libraries**

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost

**ML Techniques**

`EDA` · `Feature Engineering` · `Custom Imputation` · `Pipelines` · `Class Imbalance` · `Cross-Validation` · `RandomizedSearchCV` · `Ensemble Learning`

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

The notebook was developed in a Kaggle environment and currently expects the competition dataset through the Kaggle input path.

---

# 👨‍💻 Author

**Aayush Kotwani**

Data Science & Machine Learning Enthusiast

**Interests:** Machine Learning · Data Science · Deep Learning · Generative AI · Software Development

---

### ⭐ Key Takeaway

> **The focus of this project is not just training a model — it is building a thoughtful machine learning pipeline where data quality, feature engineering, preprocessing, model selection, and optimization all contribute to the final result.**
