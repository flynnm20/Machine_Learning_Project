import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop("Accident year", axis=1).reset_index(drop=True)
    df = df.drop("Ons code", axis=1).reset_index(drop=True)
    df = df.loc[np.repeat(df.index.values, df["Accidents"])]
    df = df.drop("Accidents", axis=1).reset_index(drop=True)
    return df


def separate_features(data_set):
    data = data_set.values
    X = data[:, 1:-1].astype(str)
    y = data[:, 0].astype(str)
    return X, y


def main():
    data_set = load_data("car-accident-data.csv")
    X, y = separate_features(data_set)
    ordinal_encoder = OrdinalEncoder()
    X = ordinal_encoder.fit_transform(X)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=1)

    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(Xtrain)
    Xtrain = ordinal_encoder.transform(Xtrain)
    Xtest = ordinal_encoder.transform(Xtest)

    label_encoder = LabelEncoder()
    label_encoder.fit(ytrain)
    ytrain = label_encoder.transform(ytrain)
    ytest = label_encoder.transform(ytest)

    model = LogisticRegression()
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    accuracy = accuracy_score(ytest, ypred)
    print('Accuracy: %.2f' % (accuracy * 100))


if __name__ == "__main__":
    main()

