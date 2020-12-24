from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt


def plot_multi_ROC(pred_prob, ytest, model_name):
    fpr = {}
    tpr = {}
    thresh = {}

    for i in range(3):
        fpr[i], tpr[i], thresh[i] = roc_curve(ytest, pred_prob[:, i], pos_label=i)

    print("Fatal AUC:" + str(metrics.auc(fpr[0], tpr[0])))
    print("Serious AUC:" + str(metrics.auc(fpr[1], tpr[1])))
    print("Slight AUC:" + str(metrics.auc(fpr[2], tpr[2])))
    # plotting
    plt.plot(fpr[0], tpr[0], color='orange', label='Fatal vs Rest')
    plt.plot(fpr[1], tpr[1], color='red', label='Serious vs Rest')
    plt.plot(fpr[2], tpr[2], color='blue', label='Slight vs Rest')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--', label="Baseline")
    plt.title('Multiclass ROC Curve ' + str(model_name))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig("Graphs/"+model_name+"_ROC_Graph")
    plt.close()
