from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# gamma for Knn gaussian calculation
from plot_multi_ROC import plot_multi_ROC

gamma = 0


def gaussian_kernel(dist):
    # w(i) = e^−γd(x^(i),x)^2
    weights = np.exp(-gamma * (dist ** 2))
    return weights / np.sum(weights)


def test_neighbors(input_prepared, output_prepared):
    kfolds = [5, 10, 20, 25, 50, 100]
    mean_errs = []
    std_errs = []
    for i in kfolds:
        kf = KFold(n_splits=i)
        tmpMeanErr = []
        for trainer, tester in kf.split(input_prepared):
            model = KNeighborsClassifier(n_neighbors=i, weights='uniform').fit(input_prepared[trainer],
                                                                               output_prepared[trainer])
            ypred = model.predict(input_prepared[tester])
            predError = f1_score(
                output_prepared[tester], ypred, average='weighted')
            tmpMeanErr.append(predError)
        mean_errs.append(np.mean(tmpMeanErr))
        std_errs.append(np.std(tmpMeanErr))

    plt.errorbar(kfolds, mean_errs, yerr=std_errs)
    plt.title('Cross validation kNN for Neighbor vales')
    plt.xlabel("Neighbor Values")
    plt.ylabel("f1-scores")
    plt.savefig("Graphs/kNN_Neigh_Crossval")
    plt.close()


def test_gamma(input_prepared, output_prepared):
    k = 5
    gamma_list = [0, 1, 5, 10, 25]
    kf = KFold(n_splits=5)
    mean_errs = []
    std_errs = []
    for i in gamma_list:
        gamma = i
        tmpMeanErr = []
        for trainer, tester in kf.split(input_prepared):
            model = KNeighborsClassifier(
                n_neighbors=k, weights=gaussian_kernel).fit(input_prepared[trainer], output_prepared[trainer])
            ypred = model.predict(input_prepared[tester])
            predError = f1_score(
                output_prepared[tester], ypred, average='weighted')
            tmpMeanErr.append(predError)
        mean_errs.append(np.mean(tmpMeanErr))
        std_errs.append(np.std(tmpMeanErr))
    plt.errorbar(gamma_list, mean_errs, yerr=std_errs)
    plt.title('Cross validation kNN for Gamma Values')
    plt.xlabel("Gamma Values")
    plt.ylabel("f1-score")
    plt.savefig("Graphs/kNN_Gamma_Crossval")
    plt.close()


def compare_tuned_model(input_data, output_data):
    # Tunned Model
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        input_data, output_data, test_size=0.33, random_state=1)
    tunned_knn_model = KNeighborsClassifier(
        n_neighbors=50, weights='uniform').fit(Xtrain, ytrain)
    tuned_model_ypred = tunned_knn_model.predict(Xtest)
    print("Tuned Knn f-1 score: " +
          str(f1_score(ytest, tuned_model_ypred, average='weighted')))

    # Dummy Model
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    dummy_classifier.fit(Xtrain, ytrain)
    dummy_classifier_ypred = dummy_classifier.predict(Xtest)
    print("Dummy Model f-1 score: " +
          str(f1_score(ytest, dummy_classifier_ypred, average='weighted')))

    pred_prob = tunned_knn_model.predict_proba(Xtest)
    plot_multi_ROC(pred_prob, ytest, 'kNN')

# does all experiments and tests tuned model against baseline model.
def knn_classification(input_data, output_data):
    # gamma test
    test_gamma(input_data, output_data)
    # resulting graph proves that gamma is irrelevant for this set

    # k test
    test_neighbors(input_data, output_data)
    # optimal k value is 50. Rather high and probably contributes to overfitting but smaller values are far less accurate
    # Increasing K tends to increase the overall accuracy because the data is so big.
    # use tuned model K = 50 and Weight is uniform.
    compare_tuned_model(input_data, output_data)