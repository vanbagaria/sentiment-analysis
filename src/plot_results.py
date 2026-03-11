import matplotlib.pyplot as plt
import pandas as pd

def plot_model_comparison(results_file="results/experiment_results.csv"):
    df = pd.read_csv(results_file)

    labels = df["model"] + "_" + df["attention"].fillna("none")

    plt.figure()

    plt.bar(labels, df["accuracy"])

    plt.xticks(rotation=45)

    plt.ylabel("Accuracy")

    plt.title("Model Accuracy Comparison")

    plt.tight_layout()

    plt.savefig("results/plots/model_comparison.png")

    plt.close()
