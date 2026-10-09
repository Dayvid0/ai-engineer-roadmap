from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans

# --- Data: 442 patients, 10 measurements each, plus a disease-progression score ---
X, y = load_diabetes(return_X_y=True)
print(f"Samples: {X.shape[0]}, features per sample: {X.shape[1]}\n")

# --- The split: 80% to learn from, 20% held back for an honest test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")

# --- SUPERVISED: regression (predict a number) ---
print("=== SUPERVISED (regression) ===")
for name, model in [
    ("Linear regression", LinearRegression()),
    ("Decision tree (no limits)", DecisionTreeRegressor(random_state=42)),
]:
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name}:")
    print(f"  train score: {train_score:.2f}")
    print(f"  test score:  {test_score:.2f}")

# --- UNSUPERVISED: clustering (no y used at all) ---
print("\n=== UNSUPERVISED (clustering) ===")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)   # note: only X, no answers given
for cluster in range(3):
    print(f"Cluster {cluster}: {(kmeans.labels_ == cluster).sum()} patients")
