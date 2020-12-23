from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

import utilities as utl
from sklearn.metrics import confusion_matrix, f1_score


def logistic_regression(xtrain, ytrain, c, is_l1):
    if is_l1:
        model = LogisticRegression(C=c, solver="saga", penalty="l1")
    else:
        model = LogisticRegression(C=c, solver="saga", penalty="l2")
    model.fit(xtrain, ytrain)
    return model


def logistic_cross_val(input_data, output_data):

    # Initialise data
    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])
    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = LabelEncoder().fit_transform(output_data)
    xtrain, xtest, ytrain, ytest = train_test_split(input_prepared, output_prepared, test_size=0.2, random_state=1)
    cs = [0.0001, 0.01, 1, 10, 1000, 100000]

    # Initialise output graph
    fig, axs = plt.subplots(2, len(cs))
    axs[0][0].set_ylabel("L1 Penalty")
    axs[1][0].set_ylabel("L2 Penalty")

    for i, c in enumerate(cs):
        # L1 penalty
        model = logistic_regression(xtrain, ytrain, c, is_l1=True)
        ypred = model.predict(xtest)
        accuracy = f1_score(ytest, ypred, average="micro")
        print('c = %.2f , L1 Penalty --> Accuracy: %.2f' % (c, accuracy * 100))
        utl.plot_roc_curves(ytest, ypred, axs[0][i])

        # L2 penalty
        model = logistic_regression(xtrain, ytrain, c, is_l1=False)
        ypred = model.predict(xtest)
        accuracy = f1_score(ytest, ypred, average="micro")
        print('c = %.2f , L2 Penalty --> Accuracy: %.2f' % (c, accuracy * 100))
        utl.plot_roc_curves(ytest, ypred, axs[1][i])
        axs[1][i].set_xlabel("C = " + str(c))
        if i != 0:
            axs[0][i].set_yticklabels([])
            axs[1][i].set_yticklabels([])
        axs[0][i].set_xticklabels([])
    fig.savefig("logistic_regression_cross_val.png", dpi=300)

