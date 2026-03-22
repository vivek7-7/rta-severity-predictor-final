# CrashSense — Road Traffic Accident Severity Predictor

A full-stack machine learning web application that predicts road accident severity
(Slight Injury / Serious Injury / Fatal Injury) using 25 features from the
Addis Ababa RTA dataset (12,316 records).

**Live Demo:** https://rta-severity-predictor-1-avcv.onrender.com

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI 0.111 (async Python) |
| Frontend | Jinja2 + Tailwind CSS + Alpine.js + Chart.js |
| Database | SQLite (dev) → PostgreSQL/Supabase (production) |
| ORM | SQLAlchemy 2.0 async |
| Auth | JWT tokens + werkzeug password hashing |
| ML | scikit-learn 1.8.0 |
| Explainability | SHAP TreeExplainer |
| Deployment | Docker + Render.com |

---

## 11 ML Models — Results (25 features)

| # | Model | Unit | Accuracy | Weighted F1 | Fatal Recall |
|---|---|---|---|---|---|
| 1 | Gradient Boosting ★ | III | 85.11% | 0.803 | 9.68% |
| 2 | XGBoost (HistGB) | III | 84.50% | 0.800 | 9.68% |
| 3 | LightGBM (HistGB) | III | 84.46% | 0.793 | 9.68% |
| 4 | Random Forest | III | 83.36% | 0.791 | 16.13% |
| 5 | KNN (k=5) | III | 83.36% | 0.779 | 0.00% |
| 6 | Naive Bayes | III | 82.51% | 0.766 | 0.00% |
| 7 | SVM (RBF) | III | 67.17% | 0.711 | 12.90% |
| 8 | Logistic Regression | III | 48.99% | 0.583 | 51.61% |
| 9 | Decision Tree | III | 47.73% | 0.555 | 45.16% |
| 10 | Ridge Regression | II | 84.58% | 0.775 | 0.00% |
| 11 | Lasso Regression | II | 84.58% | 0.775 | 0.00% |

★ Default production model — best weighted F1 score

---

## Features Used (25 of original 30)

**Removed 5 features for fairness and data integrity:**
- `Sex_of_driver` — demographic police record, no causal link to severity
- `Sex_of_casualty` — demographic police record, no causal link to severity
- `Work_of_casuality` — administrative police field
- `Fitness_of_casuality` — administrative police field
- `Casualty_severity` — data leakage (only known after accident outcome)

---

## Database — SQLite to PostgreSQL Migration

Currently uses SQLite. To switch to PostgreSQL (Supabase):

1. Create project on supabase.com
2. Copy connection string from Settings → Database → URI tab
3. Add to Render environment variables:
