import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dataset import load_imdb_dataset
from preprocess import preprocess_texts

from utils.tokenizer import (
    build_tokenizer,
    texts_to_padded_sequences
)

from models.model_builder import build_model

from evaluate import evaluate_model
from plot_training import plot_training


EXPERIMENTS = [
    ("lstm", None),
    ("gru", None),

    ("lstm", "simple"),
    ("gru", "simple"),

    ("lstm", "bahdanau"),
    ("gru", "bahdanau"),

    ("lstm", "luong"),
    ("gru", "luong")
]


def main():

    print("\nLoading dataset...")
    texts, labels = load_imdb_dataset(split="train")

    labels = np.array(labels)

    print("Dataset size:", len(texts))


    print("\nPreprocessing text...")
    texts = preprocess_texts(texts)


    print("\nSplitting dataset...")

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42
    )


    print("\nBuilding tokenizer...")

    tokenizer = build_tokenizer(X_train_texts)

    X_train = texts_to_padded_sequences(tokenizer, X_train_texts)
    X_test = texts_to_padded_sequences(tokenizer, X_test_texts)


    vocab_size = len(tokenizer.word_index) + 1
    max_len = X_train.shape[1]

    print("Vocabulary size:", vocab_size)
    print("Sequence length:", max_len)


    results = []


    for rnn_type, attention in EXPERIMENTS:

        print("\n======================================")
        print("Training model:", rnn_type, "Attention:", attention)
        print("======================================")

        model = build_model(
            vocab_size=vocab_size,
            max_len=max_len,
            rnn_type=rnn_type,
            attention=attention
        )


        history = model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=5,
            batch_size=64
        )


        metrics = evaluate_model(model, X_test, y_test)

        print("\nEvaluation Metrics")
        print("Accuracy :", metrics["accuracy"])
        print("Precision:", metrics["precision"])
        print("Recall   :", metrics["recall"])
        print("F1 Score :", metrics["f1"])


        model_name = f"{rnn_type}_{attention if attention else 'none'}"

        plot_training(history, model_name)


        results.append({
            "model": rnn_type,
            "attention": attention,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })


    results_df = pd.DataFrame(results)

    print("\nExperiment Results")
    print(results_df)


    results_df.to_csv("results/experiment_results.csv", index=False)


if __name__ == "__main__":
    main()