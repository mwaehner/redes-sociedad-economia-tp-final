import numpy as np
import matplotlib.pyplot as plt

# set width of bar
import pandas as pd

df_loss_training_version_0 = pd.read_csv(
    "..\\loss\\run-math_version_0-tag-fold_0_loss_training.csv",
    sep=",",
    usecols=['Value']
)

df_loss_validation_version_0 = pd.read_csv(
    "..\\loss\\run-math_version_0-tag-fold_0_loss_validation.csv",
    sep=",",
    usecols=['Value']
)

df_loss_training_version_1 = pd.read_csv(
    "..\\loss\\run-math_version_1-tag-fold_0_loss_training.csv",
    sep=",",
    usecols=['Value']
)

df_loss_validation_version_1 = pd.read_csv(
    "..\\loss\\run-math_version_1-tag-fold_0_loss_validation.csv",
    sep=",",
    usecols=['Value']
)

df_loss_training_version_2 = pd.read_csv(
    "..\\loss\\run-math_version_2-tag-fold_0_loss_training.csv",
    sep=",",
    usecols=['Value']
)

df_loss_validation_version_2 = pd.read_csv(
    "..\\loss\\run-math_version_2-tag-fold_0_loss_validation.csv",
    sep=",",
    usecols=['Value']
)

plt.plot(df_loss_training_version_0, label="Hidden_10")
plt.plot(df_loss_training_version_1, label="Hidden_30")
plt.plot(df_loss_training_version_2, label="Hidden_30_dropout")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training loss")

plt.legend()
plt.show()

plt.plot(df_loss_validation_version_0, label="Hidden_10")
plt.plot(df_loss_validation_version_1, label="Hidden_30")
plt.plot(df_loss_validation_version_2, label="Hidden_30_dropout")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Validation loss")

plt.legend()
plt.show()
