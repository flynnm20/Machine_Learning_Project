from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc_curves(ytest, ypred, ax):
    ytest0 = [(1 if i == 0 else 0)for i in ytest]
    ytest1 = [(1 if i == 1 else 0) for i in ytest]
    ytest2 = [(1 if i == 2 else 0) for i in ytest]
    ypred0 = [(1 if i == 0 else 0) for i in ypred]
    ypred1 = [(1 if i == 1 else 0) for i in ypred]
    ypred2 = [(1 if i == 2 else 0) for i in ypred]

    fpr0, tpr0, thresholds0 = roc_curve(ytest0, ypred0)
    ax.plot(fpr0, tpr0, color="green", label="Class 0")

    fpr1, tpr1, thresholds1 = roc_curve(ytest1, ypred1)
    ax.plot(fpr1, tpr1, color="red", label="Class 1")

    fpr2, tpr2, thresholds2 = roc_curve(ytest2, ypred2)
    ax.plot(fpr2, tpr2, color="blue", label="Class 2")
