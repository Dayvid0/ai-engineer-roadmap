import numpy as np

np.set_printoptions(precision=2, suppress=True)

def softmax(scores):
    # turn each row of scores into positive weights that sum to 1
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

tokens = ["the", "cat", "chased", "the", "mouse"]

# Hand-made 3-number "meanings" per token: [animal-ness, action-ness, article-ness]
X = np.array([
    [0.0, 0.0, 1.0],   # the
    [1.0, 0.1, 0.0],   # cat
    [0.1, 1.0, 0.0],   # chased
    [0.0, 0.0, 1.0],   # the
    [0.9, 0.0, 0.1],   # mouse
]) * 2.0               # scale up so the differences show clearly

# In a real model, Q, K, V come from three LEARNED weight matrices. Here we skip
# the learning and use X directly for all three, so the mechanics stay visible.
Q, K, V = X, X, X
d = X.shape[1]

scores = Q @ K.T / np.sqrt(d)      # how well does each token's query match every key?
weights = softmax(scores)          # each row: how much attention that token pays to every token
output = weights @ V               # each token's new vector = weighted blend of all value vectors

print("Attention weights (row = token paying attention, column = token being looked at)")
print(f"{'':8}" + "".join(f"{t:>8}" for t in tokens))
for t, row in zip(tokens, weights):
    print(f"{t:8}" + "".join(f"{w:8.2f}" for w in row))
print("\nEach row sums to:", weights.sum(axis=1))

# Causal mask: a text generator may not look at tokens that come AFTER it
mask = np.triu(np.ones((len(tokens), len(tokens))), k=1).astype(bool)
masked_scores = np.where(mask, -1e9, scores)
causal_weights = softmax(masked_scores)

print("\nWith a causal mask (what a GPT-style model does):")
print(f"{'':8}" + "".join(f"{t:>8}" for t in tokens))
for t, row in zip(tokens, causal_weights):
    print(f"{t:8}" + "".join(f"{w:8.2f}" for w in row))

# Why long context windows are expensive: every token is compared with every token
print("\nComparisons per attention layer as the context grows:")
for n in [1_000, 8_000, 128_000]:
    print(f"  {n:>7,} tokens -> {n*n:>18,} pairs")
