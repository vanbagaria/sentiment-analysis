import matplotlib.pyplot as plt

def plot_training(history, model_name):
    plt.figure()

    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")

    plt.title(f"Accuracy - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(f"results/plots/{model_name}_accuracy.png")

    plt.close()
