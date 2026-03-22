# RTA Severity Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)

A portfolio-grade full-stack ML web application that predicts road accident severity (**Slight / Serious / Fatal**) using the **Addis Ababa RTA dataset** (12,316 instances, 31 features).

Trains **12 ML algorithms** covering the full university syllabus (Units I–V), served via **FastAPI** with a beautiful **Tailwind CSS + Alpine.js + Chart.js** frontend.

---

## Features

- 🔐 **JWT Authentication** — register, login, httpOnly cookie sessions
- 🔮 **Real-time Predictions** — 31-feature accordion form → instant severity prediction
- 📊 **SHAP Explainability** — waterfall chart showing why each prediction was made
- 📈 **Dashboard** — doughnut, line, and bar charts of your prediction history
- 🗂 **Prediction History** — paginated table with filters + CSV export
- 🤖 **12 ML Models** — select any trained model for each prediction
- 📋 **Model Comparison** — sortable metrics table, confusion matrix, feature importance
- 🐳 **Docker Ready** — single-command deployment
- 🚀 **CI/CD** — GitHub Actions → Docker Hub → Render.com

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/rta-severity-predictor.git
cd rta-severity-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add dataset
mkdir data
# Download RTA_Dataset.csv from Kaggle and place it at data/RTA_Dataset.csv
# https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents

# 3. Train all 12 models (takes ~10-20 min)
python notebooks/train_all_models.py

# 4. Run the application
uvicorn app.main:app --reload --port 8000

# 5. Open http://localhost:8000
```

**Skip slow models during development:**
```bash
python notebooks/train_all_models.py --skip-slow   # skips SVM + Optuna
```

---

## Model Performance (after training)

| Rank | Model             | Unit | Accuracy | Weighted F1 | ROC-AUC | Train Time |
|------|-------------------|------|----------|-------------|---------|------------|
| 1    | XGBoost (tuned)   | III  | —        | —           | —       | —          |
| 2    | LightGBM          | III  | —        | —           | —       | —          |
| 3    | Random Forest     | III  | —        | —           | —       | —          |
| 4    | Gradient Boosting | III  | —        | —           | —       | —          |
| 5    | MLP Neural Net    | IV   | —        | —           | —       | —          |
| 6    | SVM (RBF)         | III  | —        | —           | —       | —          |
| 7    | Logistic Reg.     | III  | —        | —           | —       | —          |
| 8    | KNN (k=5)         | III  | —        | —           | —       | —          |
| 9    | Decision Tree     | III  | —        | —           | —       | —          |
| 10   | Naïve Bayes       | III  | —        | —           | —       | —          |
| 11   | Ridge Regression  | II   | —        | —           | —       | —          |
| 12   | Lasso Regression  | II   | —        | —           | —       | —          |

*Fill in after running `train_all_models.py`*

---

## Project Structure

```
rta-severity-predictor/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # Settings (pydantic-settings)
│   ├── database.py              # Async SQLAlchemy + SQLite
│   ├── models/                  # ORM: User, Prediction
│   ├── schemas/                 # Pydantic: auth, prediction
│   ├── routers/                 # auth, predict, result, history, dashboard, model_info
│   ├── ml/
│   │   ├── predictor.py         # Predictor class + SHAP + demo mode
│   │   ├── features.py          # Feature order, options, model registry
│   │   └── artifacts/           # .pkl files (gitignored)
│   └── templates/               # Jinja2 HTML (base, login, dashboard, predict, result, history, model_info)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   └── train_all_models.py      # Full training pipeline (all 12 models)
├── tests/                       # pytest suite
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/deploy.yml
```

---

## API Endpoints

| Method | Path                    | Auth | Description                     |
|--------|-------------------------|------|---------------------------------|
| GET    | /login                  | ✗    | Login / register page           |
| POST   | /login                  | ✗    | Process login → set JWT cookie  |
| POST   | /register               | ✗    | Create account                  |
| GET    | /logout                 | ✓    | Clear cookie → redirect         |
| GET    | /dashboard              | ✓    | Dashboard with charts           |
| GET    | /predict                | ✓    | 31-field prediction form        |
| POST   | /predict                | ✓    | Run prediction → redirect       |
| GET    | /result/{id}            | ✓    | Result page with SHAP           |
| GET    | /history                | ✓    | Paginated prediction history    |
| GET    | /history/export         | ✓    | Download history as CSV         |
| POST   | /history/{id}/delete    | ✓    | Delete a prediction             |
| GET    | /model-info             | ✓    | Model comparison + dataset info |
| GET    | /docs                   | ✗    | Swagger UI (auto-generated)     |
| GET    | /redoc                  | ✗    | ReDoc (auto-generated)          |

---

## Docker Deployment

```bash
# Build and run locally
docker-compose up --build

# Or manually
docker build -t rta-predictor .
docker run -p 8000:8000 \
  -e SECRET_KEY=your-secret-key \
  -v $(pwd)/app/ml/artifacts:/app/app/ml/artifacts \
  rta-predictor
```

---

## Deploy to Render.com

1. Push to GitHub main branch
2. GitHub Actions runs tests → builds Docker image → pushes to Docker Hub
3. Render deploy hook is triggered automatically
4. Set environment variables in Render dashboard:
   - `SECRET_KEY` — a long random string
   - Mount a persistent disk at `/app/app/ml/artifacts` (or pre-train and bake artifacts into image)

---

## Dataset

**Road Traffic Accidents — Addis Ababa Sub-City**  
[Kaggle Link](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents)

- 12,316 instances, 31 categorical features
- Target: `Accident_severity` → Slight Injury (75%) / Serious Injury (20%) / Fatal injury (5%)
- Class imbalance handled with **SMOTE**
- Missing values imputed with **column mode**

---

## License

MIT License — see [LICENSE](LICENSE)
