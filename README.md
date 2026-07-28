# Credit-Card-Score-Prediction<div align="center">

# 💳 Credit Card Score Prediction

### An End-to-End Machine Learning Regression Pipeline for Predicting Customer Credit Scores

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-3776AB?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](./LICENSE)

</div>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Models Used](#-models-used)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Example Prediction](#-example-prediction)
- [Visualizations](#-visualizations)
- [Configuration](#-configuration)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)
- [Why This Project Stands Out](#-why-this-project-stands-out)

---

## 📋 Project Overview

**Credit Card Score Prediction** is a supervised **regression** project that estimates a customer's numeric **credit score** from their financial and behavioral profile — age, income, employment history, credit utilization, payment history, existing debt, and credit inquiries.

> 💡 **Why it matters:** Credit scoring drives decisions across banking and fintech — loan approvals, credit limits, interest rates, and risk assessment. A reliable, data-driven scoring model helps institutions make faster, more consistent, and more explainable lending decisions.

**Real-world applications**
- 🏦 Bank & NBFC loan/credit-limit approval workflows
- 📱 Fintech and neobank onboarding risk checks
- 📊 Internal credit-risk dashboards for analysts
- 🎓 Educational reference for regression-based scoring pipelines

**Expected users:** Data science learners, credit-risk analysts, fintech developers, and recruiters evaluating applied ML/regression skills.

---

## 🎬 Demo

> 🖼️ *Add a screenshot of the Streamlit app here.*

```
![App Screenshot](assets/demo-screenshot.png)
```

> 🎥 *Add a short GIF or demo video walkthrough here.*

> 🌐 **Live Deployment:** Not implemented — the app is Docker-ready but no public hosted link is included in this repository.

---

## ✨ Features

- [x] Data cleaning & preprocessing
- [x] Categorical feature encoding (Label Encoding)
- [x] Train/test split for model validation
- [x] Multi-model training & benchmarking (10 regressors)
- [x] Quantitative evaluation (R², MAE, MSE, RMSE)
- [x] Best-model persistence with `pickle`
- [x] Interactive **Streamlit** prediction UI
- [x] **Docker**-ready for containerized deployment
- [x] Reproducible, notebook-driven workflow
- [ ] Hyperparameter tuning — *Not implemented*
- [ ] Automated CI/CD pipeline — *Not implemented*

---

## 🛠️ Tech Stack

**Languages**
- Python 3.11

**Libraries**
- Pandas, NumPy — data manipulation
- scikit-learn — preprocessing, modeling, evaluation
- XGBoost — gradient-boosted regression

**Frameworks / App Layer**
- Streamlit — interactive web UI for real-time predictions

**Deployment**
- Docker (`dockerfile` included, exposes port `8501`)

**Version Control**
- Git & GitHub

**Environment**
- Jupyter Notebook (model experimentation & training)

---

## 📂 Project Structure

```
Credit-Card-Score-Prediction/
│
├── artifact/
│   ├── credit_score.pkl          # Trained Gradient Boosting regression model
│   └── encoder.pkl               # Fitted LabelEncoder for 'payment_history'
│
├── data/
│   └── credit_score_prediction.csv   # Raw dataset (1,000 records)
│
├── notebooks/
│   └── Credit Score Prediction.ipynb # Full EDA → training → export pipeline
│
├── app.py                        # Streamlit web application
├── dockerfile                    # Container build instructions
├── requirements.txt              # Python dependencies
├── LICENSE                       # Apache 2.0 License
└── README.md                     # Project documentation
```

**Folder notes:**
- `artifact/` — serialized objects loaded directly by `app.py` at inference time.
- `data/` — source CSV consumed by the training notebook.
- `notebooks/` — single notebook covering the entire ML workflow end-to-end.

---

## 🔄 Workflow

```
Data Collection
      ↓
Data Cleaning (drop identifier column)
      ↓
Categorical Encoding (Label Encoding)
      ↓
Train/Test Split (80/20)
      ↓
Multi-Model Training & Benchmarking
      ↓
Evaluation (R², MAE, MSE, RMSE)
      ↓
Best Model Selection (Gradient Boosting)
      ↓
Model Serialization (pickle)
      ↓
Streamlit Prediction App
```

---

## 📊 Dataset

| Detail | Description |
|---|---|
| **Source** | `data/credit_score_prediction.csv` (included in repo) |
| **Samples** | 1,000 customer records |
| **Target Variable** | `credit_score` (numeric, continuous) |
| **Missing Values** | None — all 11 columns are 100% non-null |

**Feature columns:**

| Feature | Type | Description |
|---|---|---|
| `customer_id` | object | Unique identifier — dropped before training |
| `age` | int | Customer age |
| `annual_income` | int | Annual income ($) |
| `employment_years` | int | Years of employment |
| `credit_utilization` | float | Credit utilization (%) |
| `payment_history` | object → encoded | Excellent / Good / Average / Poor |
| `num_credit_cards` | int | Number of active credit cards |
| `loan_balance` | int | Outstanding loan balance ($) |
| `debt_to_income` | float | Debt-to-income ratio |
| `credit_inquiries` | int | Number of recent credit inquiries |

---

## 🔍 Exploratory Data Analysis

> The notebook performs a lightweight EDA — inspecting data types (`df.info()`), previewing samples (`df.head()`), and reviewing category distributions for `payment_history` and `num_credit_cards` via `value_counts()`.

- Dataset is fully numeric except for `payment_history` (categorical, 4 classes).
- No missing values were found across any of the 1,000 rows.
- `customer_id` carries no predictive signal and is removed prior to modeling.

> 📈 *Add distribution plots, correlation heatmaps, or box plots here to expand on EDA insights.*

---

## 🧹 Data Preprocessing

| Step | Approach |
|---|---|
| **Missing Values** | Not required — dataset contained no nulls |
| **Encoding** | `LabelEncoder` applied to `payment_history` |
| **Scaling** | Not implemented — tree-based ensemble models used, which don't require feature scaling |
| **Feature Selection** | `customer_id` dropped as a non-predictive identifier |
| **Outlier Treatment** | Not implemented |
| **Train-Test Split** | 80% train / 20% test, `random_state=42` |

---

## 🤖 Models Used

| Model | Purpose |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | Regularized baseline |
| Lasso Regression | Regularized baseline (L1) |
| Decision Tree Regressor | Non-linear benchmark |
| Random Forest Regressor | Ensemble benchmark |
| **Gradient Boosting Regressor** | **Final selected model** ✅ |
| AdaBoost Regressor | Ensemble benchmark |
| Support Vector Regressor (SVR) | Non-linear benchmark |
| K-Nearest Neighbors Regressor | Distance-based benchmark |
| XGBoost Regressor | Ensemble benchmark |

---

## 📈 Model Performance

All 10 models were trained on the same 80/20 split and evaluated on held-out test data:

| Rank | Model | R² Score | MAE | MSE | RMSE |
|---|---|---|---|---|---|
| 🥇 1 | **Gradient Boosting** | **0.9241** | **19.51** | 532.06 | **23.07** |
| 🥈 2 | Random Forest | 0.9138 | 20.72 | 603.67 | 24.57 |
| 🥉 3 | XGBoost | 0.8998 | 22.05 | 701.75 | 26.49 |
| 4 | AdaBoost | 0.8842 | 23.39 | 811.51 | 28.49 |
| 5 | Decision Tree | 0.8199 | 29.26 | 1261.69 | 35.52 |
| 6 | Linear Regression | 0.7795 | 33.15 | 1545.12 | 39.31 |
| 7 | Ridge Regression | 0.7794 | 33.16 | 1545.16 | 39.31 |
| 8 | Lasso Regression | 0.7791 | 33.22 | 1547.82 | 39.34 |
| 9 | Support Vector Regressor | -0.0071 | 69.35 | 7055.67 | 84.00 |
| 10 | K-Nearest Neighbors | -0.0433 | 70.53 | 7308.99 | 85.49 |

> ✅ **Gradient Boosting Regressor** achieved the best trade-off across all metrics and was selected as the final production model, serialized as `artifact/credit_score.pkl`.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rohitranelab/Credit-Card-Score-Prediction.git

# 2. Move into the project directory
cd Credit-Card-Score-Prediction

# 3. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

**Run locally with Streamlit:**

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

**Run with Docker:**

```bash
# Build the image
docker build -t credit_score .

# Run the container
docker run -p 8501:8501 credit_score
```

---

## 🧪 Example Prediction

Using the Streamlit interface, enter customer details such as:

| Input | Sample Value |
|---|---|
| Age | 35 |
| Annual Income | $40,248 |
| Employment Years | 17 |
| Credit Utilization | 5.8% |
| Payment History | Excellent |
| Number of Credit Cards | 1 |
| Loan Balance | $51,889 |
| Debt-to-Income | 0.33 |
| Credit Inquiries | 9 |

**Output:**

```
Predicted Credit Score: 850.00
```

---

## 🖼️ Visualizations

> 📊 *Add screenshots of the Streamlit UI, model comparison charts, or feature-importance plots here.*

```
assets/
├── app-ui.png
├── model-comparison-chart.png
└── feature-importance.png
```

---

## ⚙️ Configuration

| Parameter | Location | Default |
|---|---|---|
| Train/test split ratio | `notebooks/Credit Score Prediction.ipynb` | 80/20 |
| Random seed | `notebooks/Credit Score Prediction.ipynb` | 42 |
| Streamlit server port | `dockerfile` | 8501 |
| scikit-learn version | `requirements.txt` | 1.6.1 |

---

## 🚀 Future Improvements

- [ ] Hyperparameter tuning (GridSearchCV / Optuna) for the Gradient Boosting model
- [ ] Feature scaling & engineering for linear/distance-based models
- [ ] Model explainability with **SHAP** / **LIME**
- [ ] Cross-validation for more robust performance estimates
- [ ] CI/CD pipeline for automated testing & deployment
- [ ] Cloud deployment (Streamlit Community Cloud / AWS / Azure)
- [ ] Model monitoring & drift detection in production
- [ ] Input validation and error handling in the Streamlit form

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your code follows clean, readable formatting and includes relevant documentation.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**Rohit Rane**

- GitHub: [@Rohitranelab](https://github.com/Rohitranelab)
- Email: ranerohit996@gmail.com

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for the modeling and preprocessing toolkit
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting implementation
- [Streamlit](https://streamlit.io/) for the interactive web app framework
- The open-source Python data science community

---

## 🌟 Why This Project Stands Out

✔ End-to-end ML pipeline — from raw CSV to deployed prediction UI
✔ Benchmarked **10 regression algorithms** with transparent metric comparison
✔ Clean, minimal, production-style project structure
✔ Reproducible experiments via a single well-organized notebook
✔ Deployment-ready with a working **Streamlit + Docker** setup
✔ Honest documentation — no invented results, no overstated claims

</div>