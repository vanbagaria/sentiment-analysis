from dataset import load_imdb_dataset
from preprocess import preprocess_texts
from utils.tokenizer import build_tokenizer, texts_to_padded_sequences


texts, labels = load_imdb_dataset(split="train")

texts = preprocess_texts(texts[:500])  # small sample for test

tokenizer = build_tokenizer(texts)

X = texts_to_padded_sequences(tokenizer, texts)

print("Vocabulary size:", len(tokenizer.word_index))
print("Shape of padded data:", X.shape)

print("\nExample sequence:")
print(X[0])
