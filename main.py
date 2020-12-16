import time
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import chi2
from matplotlib import pyplot
from keras.utils import to_categorical


def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop("Accident year", axis=1).reset_index(drop=True)
    df = df.drop("Ons code", axis=1).reset_index(drop=True)
    df = df.loc[np.repeat(df.index.values, df["Accidents"])]
    df = df.drop("Accidents", axis=1).reset_index(drop=True)
    return df

def chi_sqr_test(df):
    categorical_columns = df.select_dtypes(exclude='number').drop('Accident severity', axis=1).columns

    chi2_check = []
    for i in categorical_columns:
        if chi2_contingency(pd.crosstab(df['Accident severity'], df[i]))[1] < 0.05:
            chi2_check.append('Reject Null Hypothesis')
        else:
            chi2_check.append('Fail to Reject Null Hypothesis')
    res = pd.DataFrame(data=[categorical_columns, chi2_check]).T
    res.columns = ['Column', 'Hypothesis']

    check = {}
    for i in res[res['Hypothesis'] == 'Reject Null Hypothesis']['Column']:
        dummies = pd.get_dummies(df[i])
        bon_p_value = 0.05 / df[i].nunique()
        for series in dummies:
            if chi2_contingency(pd.crosstab(df['Accident severity'], dummies[series]))[1] < bon_p_value:
                check['{}-{}'.format(i, series)] = 'Reject Null'
            else:
                check['{}-{}'.format(i, series)] = 'Fail to Reject'
    res_chi_ph = pd.DataFrame(data=[check.keys(), check.values()]).T
    res_chi_ph.columns = ['Pair', 'Hypothesis']
    print(res_chi_ph)

def logistic_regression(input_data, output_data):
    input_prepared = OneHotEncoder().fit_transform(input_data)
    output_prepared = LabelEncoder().fit_transform(output_data)
    Xtrain, Xtest, ytrain, ytest = train_test_split(input_prepared, output_prepared, test_size=0.33, random_state=1)
    model = LogisticRegression()
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    accuracy = accuracy_score(ytest, ypred)
    print('Logistic Regression --> Accuracy: %.2f' % (accuracy * 100))

def neural_net(input_data, output_data):
    input_prepared = OneHotEncoder().fit_transform(input_data)
    output_prepared = to_categorical(LabelEncoder().fit_transform(output_data))

    Xtrain, Xtest, ytrain, ytest = train_test_split(input_prepared, output_prepared, test_size=0.33, random_state=1)

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    # define the keras model
    model = Sequential()
    model.add(Dense(50, input_dim=40, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    # compile the keras model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=10, verbose=0)
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
    #chi_sqr_test(df)
    output_data = df["Accident severity"]
    input_data = df.drop(["Accident severity"], axis=1)

    logistic_regression(input_data, output_data)
    neural_net(input_data, output_data)


if __name__ == "__main__":
    main()

