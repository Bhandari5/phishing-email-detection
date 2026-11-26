import os
import re
import joblib
import nltk
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE, "figures")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024   # reject very large requests (security)

# Make sure the NLTK data is available (needed on a fresh machine)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ---------------------------------------------------------------------------
# Main model: Logistic Regression (TF-IDF + chi-square feature selection)
# ---------------------------------------------------------------------------
tfidf = joblib.load(os.path.join(BASE, "model", "tfidf.pkl"))
selector = joblib.load(os.path.join(BASE, "model", "selector.pkl"))
model = joblib.load(os.path.join(BASE, "model", "logistic_regression.pkl"))
feature_names = tfidf.get_feature_names_out()[selector.get_support()]

# Same cleaning as in the notebook
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
MAX_WORDS = 1000
MAX_CHARS = 20000
THRESHOLD = 0.5


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()[:MAX_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Model results from the notebook (shown on the "Model performance" tab)
# ---------------------------------------------------------------------------
def load_model_results():
    path = os.path.join(FIGURES, "tuned_results.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "name": r["Model"],
            "settings": r["Best settings"],
            "cv_f1": round(r["CV F1"] * 100, 2),
            "accuracy": round(r["Test Accuracy"] * 100, 2),
            "precision": round(r["Test Precision"] * 100, 2),
            "recall": round(r["Test Recall"] * 100, 2),
            "f1": round(r["Test F1"] * 100, 2),
            "main": r["Model"] == "Logistic Regression",
        })
    rows.sort(key=lambda x: not x["main"])   # main model first
    return rows


MODEL_RESULTS = load_model_results()
FIGURE_LIST = [
    ("confusion_matrices.png", "Confusion matrices", "How many emails each model classified correctly and incorrectly on the test set."),
    ("roc_curves.png", "ROC curves", "Trade-off between catching phishing and raising false alarms. Closer to the top-left is better."),
    ("lr_explainability.png", "What the main model learned", "Words with the strongest weights towards phishing and towards legitimate."),
    ("class_distribution.png", "Class distribution", "The dataset is close to balanced between phishing and legitimate emails."),
    ("email_length.png", "Email length", "Distribution of email length for each class."),
    ("top_words.png", "Most frequent words", "The most common words in phishing and legitimate emails."),
]
FIGURE_LIST = [f for f in FIGURE_LIST if os.path.exists(os.path.join(FIGURES, f[0]))]

# ---------------------------------------------------------------------------
# Extra warning signs (rule-based, shown alongside the model's prediction)
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.I)
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co/", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly")
SIGNAL_WORDS = {
    "Urgent or threatening language": ["urgent", "immediately", "suspend", "suspended", "expire", "expires",
                                       "within 24 hours", "act now", "final notice", "locked", "unauthorized",
                                       "unusual activity", "limited time"],
    "Asks for personal or login details": ["password", "login", "log in", "verify your", "confirm your",
                                           "social security", "ssn", "credit card", "bank account", "pin number",
                                           "account details", "update your"],
    "Money, prizes or offers": ["winner", "prize", "lottery", "million", "inheritance", "claim",
                                "free", "cash", "refund", "investment", "bitcoin"],
    "Generic greeting": ["dear customer", "dear user", "dear member", "dear sir", "dear friend",
                         "dear account holder", "valued customer"],
}


def find_signals(text):
    lower = text.lower()
    signals = []

    urls = URL_RE.findall(text)
    if urls:
        signals.append({"name": "Contains links", "level": "info",
                        "detail": f"{len(urls)} link(s) found. Check where they really go before clicking."})
    if IP_URL_RE.search(text):
        signals.append({"name": "Link uses a raw IP address", "level": "high",
                        "detail": "Real companies almost never link to a bare IP address."})
    short = sorted({s.rstrip("/") for s in SHORTENERS for u in urls if s in u.lower()})
    if short:
        signals.append({"name": "Shortened link", "level": "medium",
                        "detail": "Hides the real destination: " + ", ".join(short)})

    for name, words in SIGNAL_WORDS.items():
        found = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", lower)]
        if found:
            level = "high" if len(found) >= 3 else "medium"
            signals.append({"name": name, "level": level, "detail": "Found: " + ", ".join(found[:6])})

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 30:
        caps = sum(c.isupper() for c in letters) / len(letters)
        if caps > 0.3:
            signals.append({"name": "Lots of CAPITAL letters", "level": "medium",
                            "detail": f"{caps:.0%} of letters are capitals, often used to create pressure."})
    excl = text.count("!")
    if excl >= 3:
        signals.append({"name": "Many exclamation marks", "level": "low",
                        "detail": f"{excl} exclamation marks."})
    return signals


# ---------------------------------------------------------------------------
# Highlight each word of the original email with its influence
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|[A-Za-z]+", re.I)


def highlight(text, word_weights):
    segments, pos = [], 0
    for m in TOKEN_RE.finditer(text):
        if m.start() > pos:
            segments.append({"t": text[pos:m.start()]})
        tok = m.group(0)
        low = tok.lower()
        if low.startswith(("http", "www.")):
            key = "urltoken"
        elif "@" in low:
            key = "emailtoken"
        elif low in stop_words or len(low) <= 2:
            key = None
        else:
            key = lemmatizer.lemmatize(low)
        seg = {"t": tok}
        if key and word_weights.get(key):
            seg["w"] = round(float(word_weights[key]), 3)
        segments.append(seg)
        pos = m.end()
    if pos < len(text):
        segments.append({"t": text[pos:]})
    return segments


def analyse(email_text):
    cleaned = clean_text(email_text)
    vec = selector.transform(tfidf.transform([cleaned]))
    prob = float(model.predict_proba(vec)[0, 1])

    contrib = pd.Series(vec.toarray()[0] * model.coef_[0], index=feature_names)
    contrib = contrib[contrib != 0]
    top = contrib.sort_values(key=abs, ascending=False).head(12)
    unigram_weights = {k: v for k, v in contrib.items() if " " not in k}

    if prob >= 0.7:
        risk = "high"
    elif prob >= 0.3:
        risk = "medium"
    else:
        risk = "low"

    return {
        "label": "Phishing" if prob >= THRESHOLD else "Legitimate",
        "prob": round(prob * 100, 1),
        "risk": risk,
        "intercept": round(float(model.intercept_[0]), 3),
        "score": round(float(contrib.sum() + model.intercept_[0]), 3),
        "words": [{"word": w, "weight": round(float(v), 3)} for w, v in top.items()],
        "push_phishing": round(float(contrib[contrib > 0].sum()), 3),
        "push_legit": round(float(contrib[contrib < 0].sum()), 3),
        "segments": highlight(email_text, unigram_weights),
        "signals": find_signals(email_text),
        "stats": {
            "characters": len(email_text),
            "words": len(email_text.split()),
            "tokens_after_cleaning": len(cleaned.split()),
            "known_features": int(vec.nnz),
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        results=MODEL_RESULTS,
        figures=FIGURE_LIST,
        n_features=len(feature_names),
        max_chars=MAX_CHARS,
    )


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data = request.get_json(silent=True) or {}
    email_text = str(data.get("email", ""))[:MAX_CHARS]   # limit input size
    if not email_text.strip():
        return jsonify({"error": "Please paste an email first."}), 400
    return jsonify(analyse(email_text))


@app.route("/figures/<name>")
def figure(name):
    if name not in {f[0] for f in FIGURE_LIST}:
        abort(404)
    return send_from_directory(FIGURES, name)


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "That email is too large to check. Please paste a shorter email (under 100 KB)."}), 413


if __name__ == "__main__":
    app.run(debug=False)
