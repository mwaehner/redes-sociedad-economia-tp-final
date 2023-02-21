import numpy as np
import matplotlib.pyplot as plt

# set width of bar
import pandas as pd

df_acc_version_0 = pd.read_csv(
    "..\\metrics\\run-math_version_0_metrics_averaged_acc_acc-tag-metrics_averaged_acc.csv",
    sep=",",
    usecols=['Value']
)

df_precision_version_0 = pd.read_csv(
    "..\\metrics\\run-math_version_0_metrics_averaged_precision_precision-tag-metrics_averaged_precision.csv",
    sep=",",
    usecols=['Value']
)

df_recall_version_0 = pd.read_csv(
    "..\\metrics\\run-math_version_0_metrics_averaged_recall_recall-tag-metrics_averaged_recall.csv",
    sep=",",
    usecols=['Value']
)

df_f1_version_0 = pd.read_csv(
    "..\\metrics\\run-math_version_0_metrics_averaged_f1_score_f1_score-tag-metrics_averaged_f1_score.csv",
    sep=",",
    usecols=['Value']
)


df_acc_version_1 = pd.read_csv(
    "..\\metrics\\run-math_version_1_metrics_averaged_acc_acc-tag-metrics_averaged_acc.csv",
    sep=",",
    usecols=['Value']
)

df_precision_version_1 = pd.read_csv(
    "..\\metrics\\run-math_version_1_metrics_averaged_precision_precision-tag-metrics_averaged_precision.csv",
    sep=",",
    usecols=['Value']
)

df_recall_version_1 = pd.read_csv(
    "..\\metrics\\run-math_version_1_metrics_averaged_recall_recall-tag-metrics_averaged_recall.csv",
    sep=",",
    usecols=['Value']
)

df_f1_version_1 = pd.read_csv(
    "..\\metrics\\run-math_version_1_metrics_averaged_f1_score_f1_score-tag-metrics_averaged_f1_score.csv",
    sep=",",
    usecols=['Value']
)



df_acc_version_2 = pd.read_csv(
    "..\\metrics\\run-math_version_2_metrics_averaged_acc_acc-tag-metrics_averaged_acc.csv",
    sep=",",
    usecols=['Value']
)

df_precision_version_2 = pd.read_csv(
    "..\\metrics\\run-math_version_2_metrics_averaged_precision_precision-tag-metrics_averaged_precision.csv",
    sep=",",
    usecols=['Value']
)

df_recall_version_2 = pd.read_csv(
    "..\\metrics\\run-math_version_2_metrics_averaged_recall_recall-tag-metrics_averaged_recall.csv",
    sep=",",
    usecols=['Value']
)

df_f1_version_2 = pd.read_csv(
    "..\\metrics\\run-math_version_2_metrics_averaged_f1_score_f1_score-tag-metrics_averaged_f1_score.csv",
    sep=",",
    usecols=['Value']
)




plt.plot(df_acc_version_0, label="Hidden_10")
plt.plot(df_acc_version_1, label="Hidden_30")
plt.plot(df_acc_version_2, label="Hidden_30_dropout")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.title("Accuracy")

plt.legend()
plt.show()

plt.plot(df_precision_version_0, label="Hidden_10")
plt.plot(df_precision_version_1, label="Hidden_30")
plt.plot(df_precision_version_2, label="Hidden_30_dropout")
plt.ylabel("Precision")
plt.xlabel("Epoch")
plt.title("Precision")

plt.legend()
plt.show()


plt.plot(df_recall_version_0, label="Hidden_10")
plt.plot(df_recall_version_1, label="Hidden_30")
plt.plot(df_recall_version_2, label="Hidden_30_dropout")
plt.ylabel("Recall")
plt.xlabel("Epoch")
plt.title("Recall")

plt.legend()
plt.show()



plt.plot(df_f1_version_0, label="Hidden_10")
plt.plot(df_f1_version_1, label="Hidden_30")
plt.plot(df_f1_version_2, label="Hidden_30_dropout")
plt.ylabel("F1 score")
plt.xlabel("Epoch")
plt.title("F1 score")

plt.legend()
plt.show()

