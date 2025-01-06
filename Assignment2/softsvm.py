import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
import os



# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m = trainX.shape[0]
    d = trainX.shape[1]

    Id = np.eye(d)
    u = np.vstack((np.zeros((d,1)), np.ones((m,1))/m))
    v = np.vstack((np.ones((m,1)), np.zeros((m,1))))
    H = np.block([[Id, np.zeros((d,m))], [np.zeros((m,d)), np.zeros((m,m))]]) * 2 * l
    A = create_A(trainX, trainy, m, d)

    sol = solvers.qp(matrix(H), matrix(u), matrix(-A), matrix(-v))
    w = np.array(sol['x'])[:d]

    return w


def create_A(trainX, trainy, m, d):
    A = np.zeros((2 * m, d + m))
    # Fill the first m rows: [y_i * x_i, e_i.T]
    for i in range(m):
        e_i = np.zeros(m)
        e_i[i] = 1  # i-th standard basis vector in m dimensions
        # Assign the first part [y_i * x_i]
        A[i, :d] = trainy[i] * trainX[i]
        # Assign the second part [e_i.T]
        A[i, d:] = e_i

    # Fill the last m rows: [0_d, e_q.T]
    for q in range(m):
        e_q = np.zeros(m)
        e_q[q] = 1  # q-th standard basis vector in m dimensions

        # Assign the last part [0_d, e_q.T]
        A[m + q, d:] = e_q

    return A

def evaluate_svm(sample_size, l_array):
    trainX, testX, trainy, testy = load_data()

    train_results = {}
    test_results = {}

    for l in l_array:
        train_errors = []
        test_errors = []
        for _ in range(10 if sample_size==100 else 1):
            # Get a random m training examples from the training set
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:sample_size]]
            _trainy = trainy[indices[:sample_size]]
            w = softsvm(l, _trainX, _trainy)
            train_errors.append(calc_error(_trainX, _trainy, w))
            test_errors.append(calc_error(testX, testy, w))
        
        train_results[l] = {
            "mean": np.mean(train_errors),
                "min_error": np.min(train_errors),
                "max_error": np.max(train_errors)
        }
        test_results[l] = {
            "mean": np.mean(test_errors),
                "min_error": np.min(test_errors),
                "max_error": np.max(test_errors)
        }

    return train_results, test_results


def plot_results(train_results, test_results, train_results_large_sample, test_results_large_sample):
    # Create a results directory if it doesn't exist
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    # Prepare data for plotting
    l_values = list(train_results.keys())  # List of lambda values
    l_values_float = [float(l) for l in l_values]  # Convert to floats for better plotting

    train_means = [train_results[l]["mean"] for l in l_values]
    train_min_errors = [train_results[l]["min_error"] for l in l_values]
    train_max_errors = [train_results[l]["max_error"] for l in l_values]

    test_means = [test_results[l]["mean"] for l in l_values]
    test_min_errors = [test_results[l]["min_error"] for l in l_values]
    test_max_errors = [test_results[l]["max_error"] for l in l_values]

    l_values_large = list(train_results_large_sample.keys())  # List of lambda values
    l_values_large_float = [float(l) for l in l_values_large]  # Convert to floats for better plotting

    train_means_large = [train_results_large_sample[l]["mean"] for l in l_values_large]

    test_means_large = [test_results_large_sample[l]["mean"] for l in l_values_large]

    # Calculate error bars
    train_error_bars = [
        (train_means[i] - train_min_errors[i], train_max_errors[i] - train_means[i])
        for i in range(len(l_values))
    ]
    test_error_bars = [
        (test_means[i] - test_min_errors[i], test_max_errors[i] - test_means[i])
        for i in range(len(l_values))
    ]

    # Separate lower and upper error bars
    train_error_bars = np.array(train_error_bars).T  # Transpose for lower and upper
    test_error_bars = np.array(test_error_bars).T  # Transpose for lower and upper

    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot small experiment train results
    plt.errorbar(
        l_values_float, train_means, yerr=train_error_bars, fmt="-x", label="Train Error (small sample)"
    )
    
    # Plot small experiment test results
    plt.errorbar(
        l_values_float, test_means, yerr=test_error_bars, fmt="-x", label="Test Error (small sample)"
    )

    # Plot large experiment train results
    plt.errorbar(
        l_values_large_float, train_means_large, fmt='o', label="Train Error (large sample)"
    )
    
    # Plot large experiment test results
    plt.errorbar(
        l_values_large_float, test_means_large, fmt='o', label="Test Error (large sample)"
    )

    # Add labels and legend
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Error")
    plt.xscale("log")  # Use a log scale for lambda
    plt.title("Train and Test Errors with Error Bars, both experiments")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(results_dir, f"train_test_errors_both_experiments.png"))
    plt.show()
        
        
def calc_error(x,y,w):
    y_pred = np.sign(x @ w)
    error = np.mean(y_pred.flatten() != y.flatten())

    return error
     
def load_data():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    return trainX, testX, trainy, testy

def simple_test():
    # load question 2 data
    trainX, testX, trainy, testy = load_data()

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors

    #exe 1:
    simple_test()
    #exe 2:
    #Both experiment:
    plot_results(*evaluate_svm(100,[10**n for n in range(-1,12,2)]),*evaluate_svm(1000,[10**n for n in [1,3,5,8]]))
  


    
