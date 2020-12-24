from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np

from plot_multi_ROC import plot_multi_ROC


def logistic_regression(xtrain, ytrain, c):
    model = LogisticRegression(C=c, solver="saga", penalty="l2")
    model.fit(xtrain, ytrain)
    return model


def logistic_cross_val(input_prepared, output_prepared):

    # Initialise data
    cs = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    kf = KFold(n_splits=5)

    # best_model, best_score, best_pred, xtest, ytest, c = None, 0, [], [], [], 0

    mean_errs = []
    std_errs = []
    for c in cs:
        tmpMeanErr = []
        for train, test in kf.split(input_prepared):
            model = logistic_regression(input_prepared[train], output_prepared[train], c)
            ypred = model.predict(input_prepared[test])
            score = f1_score(output_prepared[test], ypred, average="weighted")
            tmpMeanErr.append(score)
        mean_errs.append(np.mean(tmpMeanErr))
        std_errs.append(np.std(tmpMeanErr))
    plt.errorbar(cs, mean_errs, yerr=std_errs)
    plt.title('Cross validation logistic regression for C Values')
    plt.xlabel("C Values")
    plt.xscale("log")
    plt.xticks(cs, ["0.0001", "0.001", "0.01", "0.1", "1", "10"])
    plt.ylabel("f1-score")
    plt.savefig("Graphs/logistic_regression_cross_val.png", dpi=300)
    plt.close()
    print("Done!")
    # print(confusion_matrix(ytest, best_pred))
    # print(classification_report(ytest, best_pred))
    # utl.print_auc(ytest, ypred)

   #  pred_prob = best_model.predict_proba(xtest)
    # plot_multi_ROC(pred_prob, ytest, 'Logistic Regression')


def tuned_logistic_regression(input_data, output_data):
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        input_data, output_data, test_size=0.2, random_state=1)
    model = logistic_regression(Xtrain, ytrain, 1)
    ypred = model.predict(Xtest)
    print("f-1 score: " +
          str(f1_score(ytest, ypred, average='weighted')))
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))

    pred_prob = model.predict_proba(Xtest)
    plot_multi_ROC(pred_prob, ytest, 'logistic regression')
