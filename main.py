import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MaxAbsScaler
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import kNN_classifier

import logistic_regression as logr


def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop("Accident year", axis=1).reset_index(drop=True)
    df = df.drop("Ons code", axis=1).reset_index(drop=True)
    df = df.loc[np.repeat(df.index.values, df["Accidents"])]
    df = df.drop("Accidents", axis=1).reset_index(drop=True)
    df = df.replace(["Unknown"], "Other")
    return df


def down_sample(df):
    df_slight = df[df["Accident severity"] == 'Slight']
    df_serious = df[df["Accident severity"] == 'Serious']
    df_minority = df[df["Accident severity"] == 'Fatal']

    df_slight_down_sampled = resample(df_slight,
                                      replace=False,
                                      n_samples=len(df_minority))

    df_serious_down_sampled = resample(df_serious,
                                       replace=False,
                                       n_samples=len(df_minority))

    df_downsampled = pd.concat(
        [df_slight_down_sampled, df_serious_down_sampled, df_minority])
    return df_downsampled



def logistic_regression(input_data, output_data):
    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = LabelEncoder().fit_transform(output_data)

    Xtrain, Xtest, ytrain, ytest = train_test_split(input_prepared, output_prepared, test_size=0.33, random_state=1)
    logr.logistic_cross_val(Xtrain, Xtest, ytrain, ytest, [0.0001, 0.01, 1, 10, 1000, 100000])


def main():
    pd.set_option('display.max_columns', None)
    df = load_data("car-accident-data.csv")
    df = down_sample(df)

    output_data = df["Accident severity"]
    input_data = df.drop(["Accident severity"], axis=1)

    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), [
         "Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    input_prepared = MaxAbsScaler().fit_transform(input_prepared)
    output_prepared = LabelEncoder().fit_transform(output_data)

    kNN_classifier.knn_classification(input_prepared, output_prepared)


if __name__ == "__main__":
    main()
