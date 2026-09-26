import tiktoken

# common encoding, close enough for estimation
encoding = tiktoken.get_encoding("cl100k_base")

text = "Peter Parker is the world's best superhero. As for me and my family we shall serve thelord "
tokens = encoding.encode(text)

print(f"Text: {text}")
print(f"Token count: {len(tokens)}")
print(f"Actual tokens: {[encoding.decode([t]) for t in tokens]}")
