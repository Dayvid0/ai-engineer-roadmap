import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

warnings.filterwarnings("ignore")

def report(name, y_true, y_pred):
    print(f"{name:34} acc={accuracy_score(y_true, y_pred):.2f}  "
          f"precision={precision_score(y_true, y_pred, zero_division=0):.2f}  "
          f"recall={recall_score(y_true, y_pred):.2f}  "
          f"F1={f1_score(y_true, y_pred):.2f}")

# ---------- Part A: a real dataset, reasonably balanced ----------
data = load_breast_cancer()
X, y = data.data, (data.target == 0).astype(int)   # 1 = malignant (the thing we want to catch)
print(f"Part A: {len(y)} tumours, {y.mean():.0%} malignant\n")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
report("Baseline (always 'benign')", y_te, baseline.predict(X_te))
report("Logistic regression", y_te, model.predict(X_te))

tn, fp, fn, tp = confusion_matrix(y_te, model.predict(X_te)).ravel()
print(f"\nConfusion matrix: caught {tp}, missed {fn}, false alarms {fp}, correct all-clears {tn}")

# ---------- Part B: the accuracy trap (synthetic, 5% positives) ----------
Xi, yi = make_classification(n_samples=5000, n_features=10, n_informative=4,
                             weights=[0.95, 0.05], class_sep=1.5, flip_y=0.01,
                             random_state=42)
print(f"\nPart B: {len(yi)} samples, {yi.mean():.0%} positive (synthetic, deliberately imbalanced)\n")
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.2, random_state=42, stratify=yi)

report("Baseline (always 'negative')", yi_te, DummyClassifier(strategy="most_frequent").fit(Xi_tr, yi_tr).predict(Xi_te))
plain = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xi_tr, yi_tr)
report("Logistic regression", yi_te, plain.predict(Xi_te))
balanced = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(Xi_tr, yi_tr)
report("Logistic + class_weight='balanced'", yi_te, balanced.predict(Xi_te))

# ---------- Part C: the threshold dial ----------
print("\nPart C: moving the decision threshold (plain logistic model)")
proba = plain.predict_proba(Xi_te)[:, 1]
print(f"{'threshold':>9} {'precision':>10} {'recall':>7} {'flagged':>8}")
for t in [0.5, 0.3, 0.2, 0.1, 0.05]:
    pred = (proba >= t).astype(int)
    print(f"{t:9.2f} {precision_score(yi_te, pred, zero_division=0):10.2f} {recall_score(yi_te, pred):7.2f} {pred.sum():8d}")
