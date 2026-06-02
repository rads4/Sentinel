# Sentinel
Explainable AI Based Phishing Website Detection System

A production-grade phishing website detection application built with
Random Forest, SHAP explainability, and a clean Flask web interface.
Designed for free-tier cloud deployment.

---

## Live Demo

Live App Deployment:
https://web-production-31aa7.up.railway.app

---

## Features

| Capability | Details |
|---|---|
| **ML Model** | Random Forest (200 trees) — 95.5% accuracy |
| **Features** | 16 URL-extractable features — no page crawling required |
| **Explainability** | SHAP TreeExplainer — exact, not approximate |
| **Charts** | Interactive Plotly.js waterfall, bar, and importance charts |
| **Plain language** | Every SHAP contribution translated to non-technical text |
| **Sample URLs** | 3 legitimate + 3 phishing examples pre-loaded |
| **Research page** | Paper metrics, methodology, feature rationale |
| **Responsive** | Works on laptop, tablet, and mobile |

---

## Quick Start (Local)

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Sentinel.git
cd Sentinel
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (one time only)

```bash
python model/train.py
```

This downloads the UCI Phishing dataset (or generates synthetic training data
if the mirror is unavailable), trains Random Forest and Logistic Regression
classifiers, and saves all artifacts to `model/artifacts/`.

Expected output:
```
=== Phishing Detector — Training Pipeline ===
[1/5] Downloading dataset...
[2/5] Selecting 16 URL-extractable features...
[3/5] Splitting data (80/20)...
[4/5] Training models...
      → Random Forest...    Accuracy: 95.5%
      → Logistic Regression Accuracy: 96.2%
[5/5] Saving artifacts...
=== Training complete ===
```

### 4. Start the application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Project Structure

```
Sentinel/
│
├── app.py                     # Flask application — all routes
│
├── features/
│   └── extractor.py           # URL feature extraction (16 features)
│
├── model/
│   ├── train.py               # Training pipeline (run once)
│   ├── predict.py             # Prediction + SHAP pipeline
│   └── artifacts/             # Generated after training
│       ├── rf_model.pkl
│       ├── lr_model.pkl
│       ├── scaler.pkl
│       ├── feature_names.json
│       └── training_report.json
│
├── templates/
│   ├── base.html              # Shared layout, nav, footer
│   ├── index.html             # Home page
│   ├── analyze.html           # URL input + sample URLs
│   ├── result.html            # Prediction + feature table
│   ├── explain.html           # SHAP dashboard
│   └── research.html          # Paper insights + model comparison
│
├── static/
│   ├── css/style.css          # Complete design system
│   └── js/
│       ├── main.js            # Nav, scroll, animations
│       ├── analyze.js         # Form, API call, loading
│       └── charts.js          # Plotly.js SHAP charts
│
├── requirements.txt
├── Procfile                   # Render deployment
├── render.yaml                # Render configuration
└── README.md
```

---

## Feature Engineering

The original paper used 30 features requiring full page crawling. This
implementation deliberately uses 16 URL-extractable features for three reasons:

1. **Prediction integrity** — no neutral placeholders for unmeasurable features
2. **SHAP honesty** — every SHAP value corresponds to something actually measured
3. **Deployment reliability** — no browser required, works on free-tier hosting

This is consistent with how production phishing detectors (e.g., Google Safe
Browsing's first-pass filter) work — URL structure is analyzed before page load.

| Feature | Category | Always Available |
|---|---|---|
| IP Address in URL | URL Structure | ✓ |
| URL Length | URL Structure | ✓ |
| URL Shortening Service | URL Structure | ✓ |
| @ Symbol in URL | URL Structure | ✓ |
| Double Slash Redirect | URL Structure | ✓ |
| Hyphen in Domain Name | Domain Analysis | ✓ |
| Subdomain Depth | Domain Analysis | ✓ |
| HTTPS Protocol | Security | ✓ |
| HTTPS Token in Domain | Security | ✓ |
| Non-Standard Port | Security | ✓ |
| Submits to Email | URL Structure | ✓ |
| Abnormal URL Structure | Domain Analysis | ✓ |
| URL Path Depth | URL Structure | ✓ |
| Suspicious TLD | Domain Analysis | ✓ |
| DNS Record | Reputation | DNS lookup (fallback: neutral) |
| Domain Age | Reputation | WHOIS (fallback: neutral) |

DNS and WHOIS lookups are attempted with short timeouts and fall back gracefully
if unavailable — the application never fails or slows noticeably due to these.

---

## API

### `POST /api/analyze`

**Request:**
```json
{ "url": "https://example.com" }
```

**Response:**
```json
{
  "url": "https://example.com",
  "domain": "example.com",
  "prediction": "Legitimate",
  "confidence": 82.3,
  "prob_phishing": 0.177,
  "prob_legitimate": 0.823,
  "risk_level": "Low",
  "features": [ ... ],
  "shap_values": [ ... ],
  "feature_names": [ ... ],
  "feature_importances": [ ... ],
  "top_explanations": [ ... ],
  "error": null
}
```

---

## Deployment

### Render Free Tier (Recommended)

1. Push this repository to GitHub.

2. Go to [render.com](https://render.com) → New → Web Service.

3. Connect your GitHub repository.

4. Render will automatically detect `render.yaml`.
   - **Build command:** `pip install -r requirements.txt && python model/train.py`
   - **Start command:** `gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT`

5. Click **Deploy**. The build takes ~3–5 minutes on first deploy.

> **Note:** The free tier spins down after 15 minutes of inactivity.
> The first request after spin-down takes ~30 seconds (model loads from disk).
> This is normal behavior — subsequent requests are fast.

### Environment Variables (Optional)

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | Flask session key | Auto-generated on Render |
| `FLASK_ENV` | `development` / `production` | `development` |
| `PORT` | Server port | `5000` |

---

## Model Performance

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest (paper) | 30 | 96.79% | 96.18% | 98.25% | 97.20% |
| Logistic Regression (paper) | 30 | 92.36% | 92.76% | 93.86% | 93.31% |
| **Random Forest (deployed)** | **16** | **~95.5%** | — | — | **~95.8%** |
| Logistic Regression (deployed) | 16 | ~96.2% | — | — | ~96.5% |

The deployed 16-feature model achieves comparable accuracy to the paper's
full 30-feature result while being faster and fully deployable without
page crawling infrastructure.

---

## Tech Stack

- **Backend:** Python 3.12, Flask 3.0
- **ML:** scikit-learn (Random Forest, Logistic Regression)
- **Explainability:** SHAP (TreeExplainer)
- **Feature Extraction:** tldextract, dnspython, python-whois
- **Frontend:** HTML5, CSS3, Vanilla JS
- **Charts:** Plotly.js 2.32
- **Fonts:** Fraunces (display), DM Sans (body), DM Mono (code)
- **Deployment:** Gunicorn, Render

---

## Disclaimer

Sentinel is an academic project built for demonstration purposes.
It is not a substitute for professional cybersecurity tools.
The simulated phishing URLs on the demo page are for illustration only — do not visit them.

---

*Built with ❤ for MCA Major Project · 2024*
