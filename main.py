import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

import logistic_regression as logr


def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop("Accident year", axis=1).reset_index(drop=True)
    df = df.drop("Ons code", axis=1).reset_index(drop=True)
    df = df.loc[np.repeat(df.index.values, df["Accidents"])]
    df = df.drop("Accidents", axis=1).reset_index(drop=True)
    return df


def main():
    pd.set_option('display.max_columns', None)
    df = load_data("car-accident-data.csv")
    output_data = df["Accident severity"]
    input_data = df.drop(["Accident severity"], axis=1)

    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = LabelEncoder().fit_transform(output_data)

    Xtrain, Xtest, ytrain, ytest = train_test_split(input_prepared, output_prepared, test_size=0.33, random_state=1)

    # model = LogisticRegression()
    # model.fit(Xtrain, ytrain)
    # ypred = model.predict(Xtest)
    # accuracy = accuracy_score(ytest, ypred)
    # print('Accuracy: %.2f' % (accuracy * 100))
    logr.logistic_cross_val(Xtrain, Xtest, ytrain, ytest, [0.1, 1, 10, 100])



if __name__ == "__main__":
    main()

