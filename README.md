# Phishing Email Detection

A machine learning prototype that classifies emails as **phishing** or **legitimate** and shows which words influenced each decision.

- **Model:** Logistic Regression on TF-IDF features (unigrams + bigrams), reduced to 5,000 features with chi-square selection
- **Dataset:** [Kaggle Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) (82,000+ emails)
- **Test results:** 98.2% accuracy, 98.3% F1

## Project files

| File | Purpose |
|------|---------|
| `phishing.ipynb` | Data exploration, cleaning, training, tuning and evaluation of four models |
| `app.py` | Flask web app that uses the trained model |
| `templates/index.html` | Web page for the app |
| `model/` | Saved TF-IDF vectoriser, feature selector and model |
| `figures/` | Charts produced by the notebook |

## Running the app

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 and paste an email to check it.

## Re-training

The dataset is not included because of its size. Download it from Kaggle, extract it into a folder called `archive (1)`, then run `phishing.ipynb`.
