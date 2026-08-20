# Supply Chain Early-Warning System

An agentic AI system that predicts which shipments are likely to be delivered
late **before they ship**, explains the risk in plain language, and lets a
multi-agent "boardroom" debate the best response action.

## What it does

- **Predicts** late-delivery risk using an XGBoost model trained on 180,000+
  real orders — using only information known *before* dispatch (no data
  leakage), so results are honest, not inflated.
- **Explains** each risky shipment in plain language via an LLM agent.
- **Debates the action** through a 4-agent boardroom (Cost, Speed, Risk,
  Sustainability agents + a Moderator), which reaches and justifies a final
  decision.
- **Answers follow-up questions** about its own reasoning.
- **Streams live orders** in real time, scored the instant they "arrive."
- **Dashboard** with filters, search, and pagination across all orders.
- **Manual scenario tool** to test any hypothetical shipment.
- **Model performance panel** showing validated accuracy metrics live.

## Tech stack

- **Backend:** Python, FastAPI, XGBoost, scikit-learn, Anthropic Claude API
- **Frontend:** React (Vite)
- **Data:** DataCo Smart Supply Chain dataset (Kaggle)

## Project structure

```
main.py                    # FastAPI backend (all API endpoints)
predict_risk.py            # Risk scoring logic
warning_agents.py          # Explanation + Action agents
boardroom_action.py        # 4-agent boardroom debate + Q&A
train_risk_model.py        # Trains the XGBoost model
evaluate.py                # Model evaluation metrics
evaluate_explanations.py   # LLM-as-judge explanation quality scoring
make_charts.py             # Generates result charts for the paper
frontend/                  # React app (Vite)
data/                      # Dataset (not included in repo — see setup)
charts/                    # Generated result figures
```

## Setup

### 1. Backend

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

Download the **DataCo Smart Supply Chain** dataset from Kaggle and place it at:
```
data/dataco_supply_chain.csv
```

### 2. Train the model (one-time)

```bash
python train_risk_model.py
```

### 3. Run the backend

```bash
uvicorn main:app --reload
```
Backend runs at `http://localhost:8000` (interactive docs at `/docs`).

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173`.

**Both the backend and frontend must be running at the same time.**

## Reproduce evaluation results

```bash
python evaluate.py                # precision, AUC, risk-tier calibration
python evaluate_explanations.py   # explanation faithfulness + clarity
python make_charts.py             # generates figures in charts/
```

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.770 |
| High-risk precision | 85.3% |
| Explanation faithfulness (LLM-judge) | 4.50 / 5 |
| Explanation clarity | 5.00 / 5 |

## Team

- **Shivam Grover** — risk prediction model, backend (FastAPI), AI agents (explanation, action, boardroom), frontend (React)
- **Suraj Venkataraman** — model evaluation, explanation quality assessment, result visualization, documentation

## CI/CD

Automated tests run on every push via GitHub Actions. See `.github/workflows/`.
