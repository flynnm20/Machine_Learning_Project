from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=33, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def neural_net(input_data, output_data):
    estimator = KerasClassifier(
        build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator,  input_data, output_data, cv=kfold)
    print("Baseline: " + (results.mean() * 100, results.std() * 100))