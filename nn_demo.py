import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")  # silence convergence warnings for a cleaner printout

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Scale the target too: networks train more smoothly when y is near 0 with spread ~1
y_mean, y_std = y_train.mean(), y_train.std()
y_train_s = (y_train - y_mean) / y_std

def evaluate(name, model):
    model.fit(X_train_s, y_train_s)
    train_pred = model.predict(X_train_s) * y_std + y_mean
    test_pred = model.predict(X_test_s) * y_std + y_mean
    n_iter = getattr(model, "n_iter_", "-")
    print(f"{name:42} {r2_score(y_train, train_pred):8.2f} {r2_score(y_test, test_pred):8.2f} "
          f"{mean_absolute_error(y_test, test_pred):7.1f} {n_iter!s:>6}")

print(f"{'model':42} {'train R2':>8} {'test R2':>8} {'MAE':>7} {'iters':>6}")
evaluate("Linear regression (the bar to beat)", LinearRegression())
evaluate("Small net: 1 layer of 8", MLPRegressor(hidden_layer_sizes=(8,), alpha=0.0, max_iter=2000, random_state=42))
evaluate("Big net, no limits: 2 layers of 256", MLPRegressor(hidden_layer_sizes=(256, 256), alpha=0.0, max_iter=2000, random_state=42))
evaluate("Big net + L2 penalty (alpha=10)", MLPRegressor(hidden_layer_sizes=(256, 256), alpha=10.0, max_iter=2000, random_state=42))
evaluate("Big net + early stopping", MLPRegressor(hidden_layer_sizes=(256, 256), alpha=0.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, max_iter=2000, random_state=42))
