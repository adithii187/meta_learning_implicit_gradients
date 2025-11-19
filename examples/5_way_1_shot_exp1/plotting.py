import pandas as pd
import matplotlib.pyplot as plt

# Load log.csv
df = pd.read_csv("log.csv")

# Create a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot losses on the first subplot
axes[0].plot(df["train_pre"], label="Train Pre")
axes[0].plot(df["train_post"], label="Train Post")
axes[0].plot(df["test_pre"], label="Test Pre")
axes[0].plot(df["test_post"], label="Test Post")
axes[0].set_xlabel("Logging Step")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss curves (before & after adaptation)")
axes[0].legend()

# Plot accuracies on the second subplot
axes[1].plot(df["train_acc"], label="Train Acc")
axes[1].plot(df["val_acc"], label="Val Acc")
axes[1].set_xlabel("Logging Step")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy curves")
axes[1].legend()

# Adjust layout so they don’t overlap
plt.tight_layout()
plt.show()
