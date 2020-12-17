from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import utilities as utl
from sklearn.metrics import confusion_matrix, accuracy_score


def logistic_regression(xtrain, ytrain, c, is_l1):
    if is_l1:
        model = LogisticRegression(C=c, solver="saga", penalty="l1")
    else:
        model = LogisticRegression(C=c, solver="saga", penalty="l2")
    model.fit(xtrain, ytrain)
    return model


def logistic_cross_val(xtrain, xtest, ytrain, ytest, cs):
    fig, axs = plt.subplots(2, len(cs))
    # fig.set_title("Logistic Regression ROC Curves")
    axs[0][0].set_ylabel("L1 Penalty")
    axs[1][0].set_ylabel("L2 Penalty")
    for i, c in enumerate(cs):
        model = logistic_regression(xtrain, ytrain, c, is_l1=True)
        ypred = model.predict(xtest)
        print(confusion_matrix(ytest, ypred))
        utl.plot_roc_curves(ytest, ypred, axs[0][i])
        model = logistic_regression(xtrain, ytrain, c, is_l1=False)
        ypred = model.predict(xtest)
        utl.plot_roc_curves(ytest, ypred, axs[1][i])
        axs[1][i].set_xlabel("C = " + str(c))
    fig.savefig("logistic_regression_cross_val.png", dpi=300)

