import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


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
    model = LogisticRegression()
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    from sklearn.metrics import f1_score
    f1_score = f1_score(ytest, ypred, average='macro')
    print(classification_report(ytest, ypred))
    print('Logistic Regression --> f-1 score: %.2f' % (f1_score * 100))


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=33, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def neural_net(input_data, output_data):
    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = to_categorical(LabelEncoder().fit_transform(output_data))

    estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator,  input_prepared, output_prepared, cv=kfold)
    print("Baseline: " + (results.mean() * 100, results.std() * 100))


def kNN(input_data, output_data):
    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), ["Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = to_categorical(LabelEncoder().fit_transform(output_data))

    Xtrain, Xtest, ytrain, ytest = train_test_split(input_prepared, output_prepared, test_size=0.33, random_state=1)
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(input_prepared, output_prepared)
    ypred = neigh.predict(Xtest)
    from sklearn.metrics import f1_score
    f1_score = f1_score(ytest, ypred, average='macro')
    print(f1_score)


def main():
    pd.set_option('display.max_columns', None)
    df = load_data("car-accident-data.csv")
    print(df['Accident severity'].value_counts(sort=True))
    df = downSample(df)
    output_data = df["Accident severity"]
    input_data = df.drop(["Accident severity"], axis=1)

    #logistic_regression(input_data, output_data)

    neural_net(input_data, output_data)


if __name__ == "__main__":
    main()
