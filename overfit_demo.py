from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("depth | train | test")
print("------|-------|------")
for depth in [1, 2, 3, 4, 5, 6, 8, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train = model.score(X_train, y_train)
    test = model.score(X_test, y_test)
    label = "none" if depth is None else str(depth)
    print(f"{label:>5} | {train:.2f}  | {test:.2f}")
