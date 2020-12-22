import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
from matplotlib import pyplot
from keras.utils import to_categorical
from sklearn.utils import resample

import logistic_regression as logr


def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop("Accident year", axis=1).reset_index(drop=True)
    df = df.drop("Ons code", axis=1).reset_index(drop=True)
    df = df.loc[np.repeat(df.index.values, df["Accidents"])]
    df = df.drop("Accidents", axis=1).reset_index(drop=True)
    return df


def downSample(df):
    df_slight = df[df["Accident severity"] == 'Slight']
    df_serious = df[df["Accident severity"] == 'Serious']
    df_minority = df[df["Accident severity"] == 'Fatal']

    df_slight_down_sampled = resample(df_slight,
                                      replace=False,
                                      n_samples=len(df_minority))

    df_serious_down_sampled = resample(df_serious,
                                       replace=False,
                                       n_samples=len(df_minority))

    df_downsampled = pd.concat([df_slight_down_sampled, df_serious_down_sampled, df_minority])
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



def neural_net(input_data, output_data):
    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = to_categorical(LabelEncoder().fit_transform(output_data))

    Xtrain, Xtest, ytrain, ytest = train_test_split(input_prepared, output_prepared, test_size=0.33, random_state=1)

    from keras.models import Sequential
    from keras.layers import Dense
    # define the keras model
    model = Sequential()
    model.add(Dense(20, input_dim=33, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    # fit model
    history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=20, verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(Xtrain, ytrain, verbose=0)
    _, test_acc = model.evaluate(Xtest, ytest, verbose=0)
    print('NN --> Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()


def main():
    pd.set_option('display.max_columns', None)
    df = load_data("car-accident-data.csv")
    df = downSample(df)
    output_data = df["Accident severity"]
    input_data = df.drop(["Accident severity"], axis=1)

    logistic_regression(input_data, output_data)
    neural_net(input_data, output_data)


if __name__ == "__main__":
    main()
