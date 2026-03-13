import os
import matplotlib.pyplot as plt

def plot_training(history, model_name):
    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure()

    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")

    plt.title(f"Accuracy - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(os.path.join(plots_dir, f"{model_name}_accuracy.png"))

    plt.close()
