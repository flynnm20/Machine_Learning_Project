import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# gamma for Knn gaussian calculation
gamma = 0


def load_data(filename):
    df = pd.read_csv(filename)
    df = df.drop("Accident year", axis=1).reset_index(drop=True)
    df = df.drop("Ons code", axis=1).reset_index(drop=True)
    df = df.loc[np.repeat(df.index.values, df["Accidents"])]
    df = df.drop("Accidents", axis=1).reset_index(drop=True)
    return df


def gaussian_kernel(dist):
    # w(i) = e^−γd(x^(i),x)^2
    weights = np.exp(-gamma*(dist**2))
    return (weights/np.sum(weights))


def ktest(input_prepared, output_prepared):
    print("Entered K test")
    kfolds = [5, 10, 20]
    mean_errs = []
    std_errs = []
    for i in kfolds:
        print("entered loop")
        gamma = i
        kf = KFold(n_splits=i)
        tmpMeanErr = []
        for trainer, tester in kf.split(input_prepared):
            print("Enter second loop")
            model = KNeighborsClassifier(
                n_neighbors=i, weights='uniform').fit(input_prepared[trainer], output_prepared[trainer])
            ypred = model.predict(input_prepared[tester])
            predError = mean_squared_error(output_prepared[tester], ypred)
            tmpMeanErr.append(predError)
        mean_errs.append(np.mean(tmpMeanErr))
        std_errs.append(np.std(tmpMeanErr))
    print(mean_errs)
    print(std_errs)
    plt.errorbar(kfolds, mean_errs, yerr=std_errs)
    plt.title('Part 2 c Matthew Flynn 17327199 Cross validation KNN')
    plt.xlabel("Values")
    plt.ylabel("Mean square error")
    plt.show()


def testGamma(input_prepared, output_prepared):
    print("Enter gamma test")
    k = 5
    gamma_list = [0, 1, 5, 10, 25]
    kf = KFold(n_splits=5)
    mean_errs = []
    std_errs = []
    for i in gamma_list:
        gamma = i
        tmpMeanErr = []
        for trainer, tester in kf.split(input_prepared):
            print("Trainer tester loop")
            model = KNeighborsClassifier(
                n_neighbors=k, weights=gaussian_kernel).fit(input_prepared[trainer], output_prepared[trainer])
            ypred = model.predict(input_prepared[tester])
            predError = mean_squared_error(output_prepared[tester], ypred)
            tmpMeanErr.append(predError)
        mean_errs.append(np.mean(tmpMeanErr))
        std_errs.append(np.std(tmpMeanErr))
    print(mean_errs)
    print(std_errs)
    plt.errorbar(gamma_list, mean_errs, yerr=std_errs)
    plt.title('Cross validation KNN: testing gamma')
    plt.xlabel("Values")
    plt.ylabel("Mean square error")
    plt.show()


# Distance metric = Euclidean
# k = [5 or 10] need to test.
# Weight = Gaussian/uniform need to test gamma values
# aggregation = Classifiaction
# need to test C as well for alpha
def knn_classification(input_prepared, output_prepared):
    # k test
    # ktest(input_prepared, output_prepared)
    # result is k = 5 is best
    # gamma test
    testGamma(input_prepared, output_prepared)


def main():
    pd.set_option('display.max_columns', None)
    df = load_data("car-accident-data.csv")
    output_data = df["Accident severity"]
    input_data = df.drop(["Accident severity"], axis=1)

    pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), [
         "Region", "Light condition", "Weather condition", "Road surface"]),
        ("ord", OrdinalEncoder(), ["Speed limit"])
    ])

    input_prepared = pipeline.fit_transform(input_data)
    output_prepared = LabelEncoder().fit_transform(output_data)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        input_prepared, output_prepared, test_size=0.33, random_state=1)

    model = LogisticRegression()
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    accuracy = accuracy_score(Ytest, Ypred)
    print('Accuracy: %.2f' % (accuracy * 100))
    knn_classification(input_prepared, output_prepared)


if __name__ == "__main__":
    main()
