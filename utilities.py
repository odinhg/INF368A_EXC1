import matplotlib.pyplot as plt

def save_train_plot(filename, train_history):
    #Plot losses and accuracies
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].plot(train_history["train_loss"], 'b', label="Train")
    axes[0].plot(train_history["val_loss"], 'g', label="Val")
    axes[1].plot(train_history["train_accuracy"], 'b', label="Train")
    axes[1].plot(train_history["val_accuracy"], 'g', label="Val")
    axes[0].title.set_text('Loss')
    axes[1].title.set_text('Accuracy')
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    plt.savefig("training_plot.png")
