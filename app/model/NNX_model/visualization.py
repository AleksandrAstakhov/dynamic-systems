import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_history, val_history, save_path="losses.png"):

    train_history = np.array(train_history)
    val_history = np.array(val_history)

    plt.figure()
    plt.plot(train_history, label="train")
    plt.plot(val_history, label="val (rollout)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    plt.close()