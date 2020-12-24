import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MaxAbsScaler
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import kNN_classifier

import logistic_regression as logr

# loads in the data and replaces some values
def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop("Accident year", axis=1).reset_index(drop=True)
    df = df.drop("Ons code", axis=1).reset_index(drop=True)
    df = df.loc[np.repeat(df.index.values, df["Accidents"])]
    df = df.drop("Accidents", axis=1).reset_index(drop=True)
    df = df.replace(["Unknown"], "Other")
    return df

# simple chi sqr test for all of the categories
def simple_chi_sqr(df):
    categorical_columns = df.select_dtypes(exclude='number').drop('Accident severity', axis=1).columns
    chi2_check = []
    for i in categorical_columns:
        if chi2_contingency(pd.crosstab(df['Accident severity'], df[i]))[1] < 0.05:
            chi2_check.append('Reject Null Hypothesis')
        else:
            chi2_check.append('Fail to Reject Null Hypothesis')
    res = pd.DataFrame(data=[categorical_columns, chi2_check]).T
    res.columns = ['Column', 'Hypothesis']
    print(res)

# downsample the data
def down_sample(df):
    df_slight = df[df["Accident severity"] == 'Slight']
    df_serious = df[df["Accident severity"] == 'Serious']
    df_minority = df[df["Accident severity"] == 'Fatal']

    # need to do it for both classes
    df_slight_down_sampled = resample(df_slight,
                                      replace=False,
                                      n_samples=len(df_minority))

    df_serious_down_sampled = resample(df_serious,
                                       replace=False,
                                       n_samples=len(df_minority))

    # combine them into a final df
    df_downsampled = pd.concat([df_slight_down_sampled, df_serious_down_sampled, df_minority])
    return df_downsampled


def main():
    # so we can see all the columns if we print
    pd.set_option('display.max_columns', None)
    # load the data and downsample
    df = load_data("car-accident-data.csv")
    df = down_sample(df)
    # make sure all of the features are significant
    simple_chi_sqr(df)
    # turn our data into input and output
    output_data = df["Accident severity"]
    input_data = df.drop(["Accident severity"], axis=1)

    # encode all of our values using a pipline
    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = LabelEncoder().fit_transform(output_data)

    # Run our models

    # logr.logistic_cross_val(input_prepared, output_prepared)
    logr.tuned_logistic_regression(input_prepared, output_prepared)
    kNN_classifier.knn_classification(input_prepared, output_prepared)


if __name__ == "__main__":
    main()
