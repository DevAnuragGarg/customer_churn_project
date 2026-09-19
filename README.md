# Customer Churn Prediction

## Github Repo: https://github.com/DevAnuragGarg/customer_churn_project


## 1. Project Overview

This project implements an end-to-end Machine Learning solution for predicting customer churn in a telecommunications business.

The business objective is to identify customers who are likely to churn so that the retention team can proactively engage with those customers.

The project uses the IBM Telco Customer Churn dataset and develops a Decision Tree classification model. Additional experiments using class imbalance handling and Random Forest are included as bonus activities.

### End-to-End Workflow

```text
Business Problem → Data Understanding → Data Preparation → EDA
→ Feature Engineering → Train/Test Split → Model Development
→ Evaluation → Interpretation → Bonus Experiments
→ Saved ML Pipeline → Flask REST API → Prediction
```

## 2. Business Problem

The objective is to predict whether a telecom customer is likely to churn. The prediction can support identification of higher-risk customers, retention campaign planning, customer segmentation, and understanding characteristics associated with churn.

### Target Variable

`Churn`

- `Yes` — customer churned
- `No` — customer did not churn

## 3. Dataset

The project uses the IBM Telco Customer Churn dataset containing **7,043 customer records** and **21 columns**.

The data contains customer demographics, tenure, subscribed services, contract information, billing information, and churn status.

The `customerID` column is removed because it is an identifier rather than a predictive customer characteristic.

## 4. Data Understanding and Preparation

The dataset was investigated using dataset shape, column names, data types, missing-value analysis, duplicate analysis, numerical/categorical feature identification, and target-variable analysis.

### Data Quality Findings

`TotalCharges` was initially stored as a string. There were **11 blank values**, corresponding to customers with zero tenure and `Churn = No`.

The column was converted to numeric and invalid values were handled as missing values and filled with `0`:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)
```

No duplicate records were identified.

## 5. Target Variable Analysis

| Churn | Percentage |
|---|---:|
| No | 73.46% |
| Yes | 26.54% |

The target is therefore imbalanced toward customers who did not churn. This is why multiple evaluation metrics were used rather than relying only on accuracy.

## 6. Exploratory Data Analysis

More than five meaningful visualizations were created.

### Churn Distribution

Approximately 73.46% did not churn and 26.54% churned.

**Business Insight:** The churned segment represents an important customer-retention consideration.

### Contract Type vs Churn

| Contract | No Churn | Churn |
|---|---:|---:|
| Month-to-month | 57.29% | 42.71% |
| One year | 88.73% | 11.27% |
| Two year | 97.17% | 2.83% |

**Business Insight:** Month-to-month customers have a substantially higher observed churn rate than customers with longer-term contracts.

### Internet Service vs Churn

| Internet Service | No Churn | Churn |
|---|---:|---:|
| DSL | 81.04% | 18.96% |
| Fiber optic | 58.11% | 41.89% |
| No internet service | 92.60% | 7.40% |

**Business Insight:** Fiber-optic customers have a higher observed churn rate than DSL customers and customers without internet service.

### Tenure Group vs Churn

| Tenure Group | No Churn | Churn |
|---|---:|---:|
| 0–12 months | 52.56% | 47.44% |
| 13–24 months | 71.29% | 28.71% |
| 25–48 months | 79.61% | 20.39% |
| 49–72 months | 90.49% | 9.51% |

**Business Insight:** Newer customers have higher observed churn rates, while churn decreases as tenure increases.

### Monthly Charges vs Churn

| Churn | Average | Median |
|---|---:|---:|
| No | 61.27 | 64.43 |
| Yes | 74.44 | 79.65 |

**Business Insight:** Customers who churn generally have higher monthly charges in this dataset.

### Tenure vs Monthly Charges

A scatter plot examined tenure, monthly charges, and churn.

**Business Insight:** Churned customers are more concentrated among lower-tenure customers, while long-tenure customers are predominantly non-churners.

These observations describe associations in the dataset and do not establish causation.

## 7. Feature Engineering

Two meaningful features were created.

### ServiceCount

Counts the number of additional services subscribed to from OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and StreamingMovies.

**Why useful:** It summarizes additional-service adoption and may capture customer engagement.

### IsNewCustomer

```python
df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)
```

- `1` — tenure <= 12 months
- `0` — tenure > 12 months

**Why useful:** EDA showed higher observed churn among newer customers.

## 8. Train/Test Split

A 70:30 split was used with `random_state=42`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
```

| Dataset | Records |
|---|---:|
| Training | 4,930 |
| Testing | 2,113 |

## 9. Data Preprocessing

Numerical features:

- `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `ServiceCount`, `IsNewCustomer`

Categorical features:

- `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

Categorical features were transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing transformer was fitted only on training data to avoid data leakage. The 21 input features became **47 processed features** after encoding.

## 10. Model Development

The required model was a Decision Tree Classifier. Two configurations were evaluated.

### Model 1 — Default

```python
DecisionTreeClassifier(random_state=42)
```

### Model 2 — Controlled

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
```

The second configuration restricts tree complexity to reduce overfitting.

## 11. Decision Tree Comparison

### Model 1

| Metric | Score |
|---|---:|
| Accuracy | 73.73% |
| Precision | 51.75% |
| Recall | 48.95% |
| F1 Score | 50.31% |

Confusion Matrix:

```text
[[1277, 262],
 [ 293, 281]]
```

### Model 2

| Metric | Score |
|---|---:|
| Accuracy | 79.37% |
| Precision | 62.06% |
| Recall | 61.85% |
| F1 Score | 61.95% |

Confusion Matrix:

```text
[[1322, 217],
 [ 219, 355]]
```

Model 2 was selected for the final pipeline because it improved the evaluated test metrics and showed substantially better generalization.

## 12. Overfitting Analysis

Model 1 training accuracy was **99.84%** versus test accuracy of **73.73%**, a gap of approximately **26.11 percentage points**.

Model 2 training accuracy was **80.28%** versus test accuracy of **79.37%**, a gap of approximately **0.92 percentage points**.

The much smaller gap for Model 2 indicates better generalization on the test data.

## 13. Model Evaluation

The selected Decision Tree was evaluated using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

- **Accuracy:** 79.37%
- **Precision:** 62.06%
- **Recall:** 61.85%
- **F1 Score:** 61.95%

A false negative is an actual churner predicted as `No`, representing a missed potential retention opportunity. A false positive is a non-churner predicted as `Yes`, potentially resulting in unnecessary retention effort.

Recall is important for identifying actual churners, while precision is also relevant because retention activities require resources.

## 14. Feature Importance

Top features from the final Decision Tree:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | Contract — Month-to-month | 0.5070 |
| 2 | OnlineSecurity — No | 0.1403 |
| 3 | tenure | 0.1102 |
| 4 | InternetService — DSL | 0.0782 |
| 5 | TotalCharges | 0.0629 |
| 6 | MonthlyCharges | 0.0480 |
| 7 | Contract — Two year | 0.0189 |
| 8 | PaymentMethod — Electronic check | 0.0176 |

Contract type is the strongest feature used by the trained Decision Tree. Feature importance represents the contribution of encoded features to tree impurity reduction and should not be interpreted as causal impact.

## 15. Decision Tree Interpretation

The root split is based on:

```text
Contract_Month-to-month
```

Further splits include `MonthlyCharges`, `OnlineSecurity`, `tenure`, and `InternetService`.

The visualization uses Gini impurity. Lower Gini impurity represents a purer node.

## 16. Bonus 1 — Handling Class Imbalance

The target variable contains 73.46% `No` and 26.54% `Yes`. A balanced Decision Tree was tested using:

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)
```

### Results

| Metric | Model 2 | Balanced Decision Tree |
|---|---:|---:|
| Accuracy | 79.37% | 74.59% |
| Precision | 62.06% | 52.05% |
| Recall | 61.85% | 81.71% |
| F1 Score | 61.95% | 63.59% |

Confusion Matrix:

```text
[[1107, 432],
 [ 105, 469]]
```

### Analysis

Recall increased from 61.85% to 81.71%, and false negatives decreased from 219 to 105. However, false positives increased from 217 to 432. Precision decreased from 62.06% to 52.05%, and accuracy decreased from 79.37% to 74.59%. The F1 score increased slightly from 61.95% to 63.59%.

### Outcome

Class balancing demonstrates a clear precision-recall trade-off. It improves identification of the minority churn class but also increases false-positive predictions. The balanced model is retained as a bonus experiment and does not replace the deployed Decision Tree pipeline.

## 17. Bonus 2 — Random Forest

A Random Forest was tested as an additional classification model using the same training/test split and preprocessed features.

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Results

| Metric | Model 2 Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 79.37% | 78.75% |
| Precision | 62.06% | 65.21% |
| Recall | 61.85% | 46.69% |
| F1 Score | 61.95% | 54.42% |

Confusion Matrix:

```text
[[1396, 143],
 [ 306, 268]]
```

### Analysis

The Random Forest produces fewer false positives than Model 2, resulting in higher precision. However, it produces more false negatives, so fewer actual churners are identified. Its recall and F1 score are lower than those of Model 2 in this configuration.

### Outcome

The Random Forest demonstrates a different precision-recall trade-off from the Decision Tree. It is included as an additional model experiment and does not replace the deployed Decision Tree pipeline.

## 18. Final Model

The final deployed model is:

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
```

Final test performance:

| Metric | Result |
|---|---:|
| Accuracy | 79.37% |
| Precision | 62.06% |
| Recall | 61.85% |
| F1 Score | 61.95% |

## 19. End-to-End ML Pipeline

The final saved pipeline contains:

```text
Raw Customer Data
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
One-Hot Encoding
        ↓
Decision Tree Classifier
        ↓
Prediction
```

Feature engineering includes `customerID` removal, `TotalCharges` conversion/handling, `ServiceCount`, and `IsNewCustomer`.

Keeping these steps together ensures consistent transformations during training and prediction.

## 20. Model Persistence

The complete pipeline is saved using Joblib:

```python
joblib.dump(final_pipeline, "model/churn_model.pkl")
```

Saved artifact:

```text
model/churn_model.pkl
```

The artifact contains feature engineering, preprocessing, and the Decision Tree classifier.

## 21. REST API

The Flask API is implemented in `app.py`.

### Endpoint

```text
POST /predict
```

The endpoint accepts JSON customer information, validates required fields, uses the saved pipeline, generates a churn prediction, calculates the model-estimated churn probability, and returns JSON.

## 22. API Input

Required prediction fields:

```text
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
```

`customerID` is not required for prediction.

## 23. Sample API Request

The sample request is stored in `sample_request.json`:

```json
{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0
}
```

## 24. Sample API Response

Tested response:

```json
{
    "churn_prediction": "Yes",
    "churn_probability": 0.6782
}
```

The probability is the probability assigned by the trained model to the predicted `Yes` class.

## 25. Setup and Execution

### Prerequisites

- Python 3
- pip
- Jupyter Notebook/JupyterLab
- Git (optional)

### Step 1 — Create a virtual environment

From the project root:

```powershell
python -m venv .venv
```

### Step 2 — Activate the environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, use Command Prompt:

```cmd
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Run the notebook

```powershell
jupyter notebook
```

Open:

```text
notebook/churn_analysis.ipynb
```

Run the notebook from top to bottom. It performs data understanding, cleaning, EDA, feature engineering, train/test splitting, preprocessing, Decision Tree modelling, evaluation, interpretation, bonus experiments, pipeline creation, and model saving.

### Step 5 — Run the Flask API

From the project root:

```powershell
python app.py
```

The API runs at:

```text
http://127.0.0.1:5000
```

### Step 6 — Test `/predict`

```powershell
$body = Get-Content .\sample_request.json -Raw

Invoke-RestMethod `
    -Uri http://127.0.0.1:5000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The request must use `Content-Type: application/json`.

## 26. API Validation

The API checks that all required fields are present. Missing fields result in an HTTP 400 response.

Example:

```json
{
    "error": "Missing required fields",
    "missing_fields": [
        "TotalCharges"
    ]
}
```

## 27. Project Structure

```text
customer_churn_project/
│
├── data/
│   └── TelcoCustomerChurn.csv  
│
├── docs/
│   └── assignment PDFs
│
├── model/
│   └── churn_model.pkl
│
├── notebook/
│   └── churn_analysis.ipynb
│
├── app.py
├── feature_engineering.py
├── README.md
├── requirements.txt
└── sample_request.json
```

### File Descriptions

| File | Description |
|---|---|
| `notebook/churn_analysis.ipynb` | Complete analysis, EDA, modelling, evaluation, interpretation, and bonus experiments |
| `data/TelcoCustomerChurn.csv` | Telco customer churn dataset |
| `data/TelcoCustomerChurn - Data Dictionary.csv` | Dataset data dictionary |
| `feature_engineering.py` | Reusable feature engineering and cleaning logic |
| `model/churn_model.pkl` | Saved end-to-end ML pipeline |
| `app.py` | Flask REST API |
| `sample_request.json` | Sample `/predict` request |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation and setup instructions |

## 28. Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Flask
- Jupyter Notebook
- VS Code

## 29. Assignment Requirement Coverage

| Requirement | Status |
|---|---|
| Data understanding and preparation | Complete |
| Data type/structure checks | Complete |
| Missing-value analysis | Complete |
| Duplicate analysis | Complete |
| Numerical/categorical identification | Complete |
| Target analysis | Complete |
| Data cleaning/preprocessing | Complete |
| Categorical encoding | Complete |
| 70:30 train/test split | Complete |
| `random_state=42` | Complete |
| Leakage-aware preprocessing | Complete |
| At least 5 EDA visualizations | Complete |
| Business insights | Complete |
| At least 2 engineered features | Complete |
| Two Decision Tree configurations | Complete |
| Model comparison | Complete |
| Accuracy/Precision/Recall/F1 | Complete |
| Confusion Matrix | Complete |
| Feature importance | Complete |
| Decision Tree interpretation | Complete |
| Saved model/pipeline | Complete |
| Flask REST API | Complete |
| `POST /predict` | Complete |
| JSON input/output | Complete |
| Invalid input handling | Complete |
| Bonus — Class imbalance | Complete |
| Bonus — Random Forest | Complete |

## 30. Final Submission Checklist

- [ ] Jupyter Notebook
- [ ] Dataset
- [ ] Data dictionary
- [ ] `feature_engineering.py`
- [ ] `app.py`
- [ ] `model/churn_model.pkl`
- [ ] `requirements.txt`
- [ ] `README.md`
- [ ] `sample_request.json`
- [ ] Successful `/predict` API test
- [ ] Invalid-input API test
- [ ] Bonus 1 — Class Imbalance
- [ ] Bonus 2 — Random Forest

## 31. Conclusion

This project demonstrates a complete customer churn prediction workflow from raw data through data preparation, EDA, feature engineering, model development, evaluation, interpretation, model persistence, and REST API deployment.

The final deployed pipeline uses the controlled Decision Tree and achieves **79.37% test accuracy**, **62.06% precision**, **61.85% recall**, and **61.95% F1 score**.

The project also includes two bonus experiments: class imbalance handling using `class_weight="balanced"` and a Random Forest classifier.
