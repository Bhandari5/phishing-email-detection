import os
import re
import joblib
import nltk
import pandas as pd
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024   # reject very large requests (security)

# Load the trained model files
tfidf = joblib.load(os.path.join(BASE, "model", "tfidf.pkl"))
selector = joblib.load(os.path.join(BASE, "model", "selector.pkl"))
model = joblib.load(os.path.join(BASE, "model", "logistic_regression.pkl"))
feature_names = tfidf.get_feature_names_out()[selector.get_support()]

# Make sure the NLTK data is available (needed on a fresh machine)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Same cleaning as in the notebook
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
MAX_WORDS = 1000
MAX_CHARS = 20000

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()[:MAX_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    email_text = ""
    if request.method == "POST":
        email_text = request.form.get("email", "")[:MAX_CHARS]   # limit input size
        if email_text.strip():
            vec = selector.transform(tfidf.transform([clean_text(email_text)]))
            prob = model.predict_proba(vec)[0, 1]
            contrib = pd.Series(vec.toarray()[0] * model.coef_[0], index=feature_names)
            contrib = contrib[contrib != 0].sort_values(key=abs, ascending=False).head(8)
            result = {
                "label": "Phishing" if prob >= 0.5 else "Legitimate",
                "prob": round(prob * 100, 1),
                "words": [{"word": w, "weight": round(float(v), 3)} for w, v in contrib.items()],
            }
    return render_template("index.html", result=result, email_text=email_text)

@app.errorhandler(413)
def too_large(e):
    error = "That email is too large to check. Please paste a shorter email (under 100 KB)."
    return render_template("index.html", result=None, email_text="", error=error), 413

if __name__ == "__main__":
    app.run(debug=False)