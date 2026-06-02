"""
Prediction module.
Loads trained artifacts, runs feature extraction, generates
SHAP explanations, and returns a structured result dict.
"""

import os
import json
import joblib
import numpy as np
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(BASE_DIR, "model", "artifacts")

# ---------------------------------------------------------------------------
# Plain-language SHAP explanation templates
# Each entry: (feature_name, phishing_direction, phishing_text, legit_text)
# phishing_direction: True  = high value → phishing (i.e. SHAP contribution > 0 means phishing)
# ---------------------------------------------------------------------------
PLAIN_LANGUAGE = {
    "having_ip_address": {
        -1: "The URL uses a raw IP address instead of a domain name. Legitimate sites virtually never do this.",
        0:  "No IP address detected in the URL.",
        1:  "The URL uses a proper domain name, as expected for legitimate sites.",
    },
    "url_length": {
        -1: "The URL is unusually long. Phishing URLs are often padded with extra characters to obscure the real destination.",
        0:  "The URL length is in a moderately suspicious range.",
        1:  "The URL is a normal length, consistent with legitimate websites.",
    },
    "shortening_service": {
        -1: "A URL shortening service was detected. These hide the real destination and are a common phishing tactic.",
        0:  "No URL shortening service detected.",
        1:  "No URL shortening service detected — the destination is visible.",
    },
    "having_at_symbol": {
        -1: "The URL contains an @ symbol. Browsers ignore everything before @ in a URL, which attackers use to disguise the real domain.",
        0:  "No @ symbol in URL.",
        1:  "No @ symbol — the URL is straightforward.",
    },
    "double_slash_redirect": {
        -1: "A double slash (//) was found in the URL path. This is used to redirect browsers to a completely different site.",
        0:  "No double slash redirect detected.",
        1:  "No redirect tricks detected in the URL path.",
    },
    "prefix_suffix_hyphen": {
        -1: "The domain name contains a hyphen. Attackers use hyphens to imitate real brands (e.g., paypal-secure.com).",
        0:  "No hyphen found in domain.",
        1:  "No hyphens in the domain name — consistent with legitimate branding.",
    },
    "subdomain_depth": {
        -1: "The URL has multiple subdomains. Phishing sites use this to embed a trusted name early in the URL (e.g., paypal.evil.com).",
        0:  "The URL has a single subdomain — borderline.",
        1:  "The URL has no unusual subdomains.",
    },
    "https_present": {
        -1: "The URL uses HTTP, not HTTPS. No encryption is present — a significant red flag.",
        0:  "HTTPS status is unclear.",
        1:  "The URL uses HTTPS with a valid security connection.",
    },
    "https_in_domain": {
        -1: "The word 'https' appears inside the domain name itself. This is a trick to make insecure URLs look safe.",
        0:  "No HTTPS embedding trick detected.",
        1:  "No HTTPS deception detected in the domain name.",
    },
    "non_standard_port": {
        -1: "A non-standard port is specified in the URL. Legitimate websites use standard ports (80, 443).",
        0:  "Standard port in use.",
        1:  "Standard port used — consistent with legitimate web traffic.",
    },
    "submitting_to_email": {
        -1: "The URL submits data directly to an email address. This is used to harvest credentials.",
        0:  "No email submission detected.",
        1:  "No email submission detected.",
    },
    "abnormal_url": {
        -1: "The domain name doesn't match the expected registered domain. This is a structural anomaly common in phishing.",
        0:  "Domain structure is ambiguous.",
        1:  "The domain structure is consistent and normal.",
    },
    "url_depth": {
        -1: "The URL path is unusually deep. Phishing pages often use long paths to mimic legitimate site structures.",
        0:  "URL path depth is moderate.",
        1:  "The URL path depth is normal.",
    },
    "suspicious_tld": {
        -1: "The top-level domain (TLD) is on a list of domains frequently associated with phishing campaigns.",
        0:  "TLD is uncommon but not definitively suspicious.",
        1:  "The TLD is well-established and commonly used by legitimate organizations.",
    },
    "dns_record": {
        -1: "No DNS record was found for this domain. The domain may not exist or may have been newly registered.",
        0:  "DNS lookup was inconclusive or timed out.",
        1:  "A valid DNS record was found — the domain is registered and resolves correctly.",
    },
    "domain_age": {
        -1: "The domain appears to be newly registered. Phishing sites are often created just before an attack.",
        0:  "Domain age could not be determined.",
        1:  "The domain has been registered for over a year — a strong indicator of legitimacy.",
    },
}

# Risk level thresholds (phishing probability)
def _risk_level(prob_phishing: float) -> str:
    if prob_phishing < 0.35:
        return "Low"
    if prob_phishing < 0.65:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Model loader (singleton pattern — loaded once per process)
# ---------------------------------------------------------------------------
_cache = {}

def _load_artifacts():
    global _cache
    if _cache:
        return _cache

    rf     = joblib.load(os.path.join(ARTIFACTS, "rf_model.pkl"))
    lr     = joblib.load(os.path.join(ARTIFACTS, "lr_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS, "scaler.pkl"))

    with open(os.path.join(ARTIFACTS, "feature_names.json")) as f:
        feature_names = json.load(f)

    # Build SHAP explainer for RF
    # Use TreeExplainer — exact, fast, no sampling needed
    rf_explainer = shap.TreeExplainer(rf)

    _cache = {
        "rf": rf,
        "lr": lr,
        "scaler": scaler,
        "feature_names": feature_names,
        "rf_explainer": rf_explainer,
    }
    return _cache


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def _shap_for_instance(explainer, x_scaled, feature_names):
    """
    Returns SHAP values for the phishing class (index 1) for one instance.
    Shape: (n_features,)
    """
    sv = explainer.shap_values(x_scaled)
    sv = np.array(sv)
    # Newer SHAP returns shape (n_samples, n_features, n_classes)
    if sv.ndim == 3:
        vals = sv[0, :, 1]   # phishing class
    elif sv.ndim == 2 and isinstance(explainer.shap_values(x_scaled), list):
        vals = sv[1][0]
    elif sv.ndim == 2:
        vals = sv[0]
    else:
        vals = sv.flatten()
    return vals.tolist()


def _build_plain_explanations(shap_vals, feature_vals, feature_names, top_n=5):
    """
    Returns top_n features sorted by |SHAP| with plain-language text.
    """
    shap_list = [float(v) for v in shap_vals]
    indexed = sorted(range(len(shap_list)), key=lambda i: abs(shap_list[i]), reverse=True)[:top_n]
    indexed = [(i, shap_list[i]) for i in indexed]

    explanations = []
    for idx, sv in indexed:
        name = feature_names[idx]
        raw_val = int(feature_vals[idx])
        plain_map = PLAIN_LANGUAGE.get(name, {})
        text = plain_map.get(raw_val, f"Feature '{name}' value: {raw_val}")

        direction = "phishing" if sv > 0 else "legitimate"
        explanations.append({
            "feature": name,
            "shap_value": round(float(sv), 4),
            "feature_value": raw_val,
            "direction": direction,
            "plain_text": text,
        })
    return explanations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(url: str) -> dict:
    """
    Full prediction pipeline for a single URL.

    Returns a dict containing:
      - url
      - prediction: "Phishing" | "Legitimate"
      - confidence: 0–100 (probability of predicted class)
      - prob_phishing: 0.0–1.0
      - prob_legitimate: 0.0–1.0
      - risk_level: "Low" | "Medium" | "High"
      - features: list of feature detail dicts
      - shap_values: list of floats (for Plotly chart)
      - feature_names: list of strings
      - top_explanations: list of plain-language explanation dicts
      - error: None | str
    """
    from features.extractor import extract_features

    # 1. Extract features
    try:
        feat_result = extract_features(url)
    except Exception as e:
        return {"error": str(e)}

    x_raw = np.array(feat_result["vector"]).reshape(1, -1)

    # 2. Load artifacts
    try:
        arts = _load_artifacts()
    except Exception as e:
        return {"error": f"Model artifacts not found. Run: python model/train.py\n{e}"}

    # 3. Scale
    x_scaled = arts["scaler"].transform(x_raw)

    # 4. Predict
    proba = arts["rf"].predict_proba(x_scaled)[0]  # [prob_legit, prob_phishing]
    prob_legit    = float(proba[0])
    prob_phishing = float(proba[1])
    predicted_class = 1 if prob_phishing >= 0.5 else 0
    prediction = "Phishing" if predicted_class == 1 else "Legitimate"
    confidence = round((prob_phishing if predicted_class == 1 else prob_legit) * 100, 1)

    # 5. SHAP
    try:
        shap_vals = _shap_for_instance(
            arts["rf_explainer"], x_scaled, arts["feature_names"]
        )
    except Exception:
        shap_vals = np.zeros(len(arts["feature_names"]))

    # 6. Plain-language explanations
    top_explanations = _build_plain_explanations(
        shap_vals, feat_result["vector"], arts["feature_names"], top_n=6
    )

    # 7. Global feature importance (for summary bar chart)
    fi = arts["rf"].feature_importances_.tolist()

    return {
        "url": url,
        "domain": feat_result["domain"],
        "prediction": prediction,
        "confidence": confidence,
        "prob_phishing": round(prob_phishing, 4),
        "prob_legitimate": round(prob_legit, 4),
        "risk_level": _risk_level(prob_phishing),
        "features": feat_result["details"],
        "shap_values": [round(float(v), 4) for v in shap_vals],
        "feature_names": arts["feature_names"],
        "feature_importances": fi,
        "top_explanations": top_explanations,
        "error": None,
    }


def get_training_report() -> dict:
    """Returns training metrics for the Research Insights page."""
    path = os.path.join(ARTIFACTS, "training_report.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
