from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Choose depth using ONLY the training data (5-fold cross-validation)
print("depth | cross-val score (train data only)")
best_depth, best_score = None, -1
for depth in [1, 2, 3, 4, 5, 6, 8, 10]:
    scores = cross_val_score(
        DecisionTreeRegressor(max_depth=depth, random_state=42),
        X_train, y_train, cv=5
    )
    avg = scores.mean()
    print(f"{depth:>5} | {avg:.2f}")
    if avg > best_score:
        best_depth, best_score = depth, avg

print(f"\nChosen depth: {best_depth}")

# Test set touched ONCE, at the very end, for both finalists
tree = DecisionTreeRegressor(max_depth=best_depth, random_state=42).fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)
print(f"\nFinal test score - tree (depth {best_depth}): {tree.score(X_test, y_test):.2f}")
print(f"Final test score - linear regression:  {linear.score(X_test, y_test):.2f}")
