import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = {
        "k": k,
        "x_train": x_train,
        "y_train": y_train,
    }
    return classifier

def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k, x_train, y_train = classifier["k"], classifier["x_train"], classifier["y_train"]
    y_pred = []


    for test_point in x_test:
        distances = np.linalg.norm(x_train - test_point, axis=1)
        k_neighbors = y_train[np.argsort(distances)[:k]]
        y_pred.append(np.bincount(k_neighbors.astype(int)).argmax())

    return np.array(y_pred).reshape(x_test.shape[0], 1)

def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)
    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")

def corrupt_labels(y, test_classes, corruption_rate=0.3):
    """
    Randomly corrupt a portion of labels.

    
    :param y: (numpy.ndarray): Original labels.
    :param num_classes: (int): Total number of unique classes.
    :param corruption_rate: (float): Fraction of labels to corrupt.

    Returns:
        numpy.ndarray: Labels with corruption applied.
    """
    y_corrupted = y.copy()
    num_to_corrupt = int(len(y) * corruption_rate)
    indices_to_corrupt = np.random.choice(len(y), num_to_corrupt, replace=False)

    for idx in indices_to_corrupt:
        current_label = y[idx]
        y_corrupted[idx] = np.random.choice([label for label in test_classes if label != current_label])

    return y_corrupted


def evaluate_knn(k_values, train_data, train_classes, test_data, sample_sizes=None, sample_repeat=10, corrupt=False):
    x_test, y_test = test_data
    results = {}

    for k in k_values:
        for sample_size in sample_sizes or [sum(len(data) for data in train_data)]:
            errors = []
            for _ in range(sample_repeat):
                x_train, y_train = gensmallm(train_data, train_classes, sample_size)

                if corrupt:
                    y_train = corrupt_labels(y_train, train_classes)
                    y_test_corrupted = corrupt_labels(y_test, train_classes)
                else:
                    y_test_corrupted = y_test

                classifier = learnknn(k, x_train, y_train)
                y_pred = predictknn(classifier, x_test)
                errors.append(np.mean(y_pred != np.vstack(y_test_corrupted)))

            results[(k, sample_size)] = {
                "mean": np.mean(errors),
                "min_error": np.min(errors),
                "max_error": np.max(errors)
            }

    return results

def plot_results(results, x_label, y_label, title, group_by=None):
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    grouped_results = defaultdict(list)
    for (k, sample_size), values in results.items():
        key = k if group_by == 'k' else sample_size
        grouped_results[key].append(values)

    # Extract data for plotting
    x_values = sorted(grouped_results.keys())
    mean_errors = [np.mean([res["mean"] for res in grouped_results[x]]) for x in x_values]
    min_errors = [np.mean([res["min_error"] for res in grouped_results[x]]) for x in x_values]
    max_errors = [np.mean([res["max_error"] for res in grouped_results[x]]) for x in x_values]

    lower_errors = [mean - min_e for mean, min_e in zip(mean_errors, min_errors)]
    upper_errors = [max_e - mean for mean, max_e in zip(mean_errors, max_errors)]

    # Plot the results
    plt.errorbar(x_values, mean_errors, yerr=[lower_errors, upper_errors], fmt='o', capsize=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    sanitized_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    file_path = os.path.join(results_dir, f"{sanitized_title}.png")
    plt.savefig(file_path, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # setup 
    data = np.load('mnist_all.npz')
    classes = [2,3,5,6]
    train_data = [data[f'train{i}'] for i in classes]
    test_data = [data[f'test{i}'] for i in classes]
    x_test, y_test = gensmallm(test_data, classes, sum(len(test) for test in test_data))
    k_values = range(1,12)

    # Run algorithm
    print('Running question 2.a:')
    different_sample_size_results = evaluate_knn([1],train_data, classes, (x_test, y_test), range(1, 101, 10))
    plot_results(different_sample_size_results, "Sample Size", "Mean Error", title="NN Error vs. Sample size")
    print('Question 2.a Done')

    print('Running question 2.c:')
    different_k_results = evaluate_knn(k_values,train_data, classes, (x_test, y_test), sample_sizes=[200])
    plot_results(different_k_results, "K", "Mean Error", group_by='k', title="k-NN Error vs. K")
    print('Question 2.c Done')

    print('Running question 2.e:')
    results_with_corruption = evaluate_knn(k_values, train_data, classes, (x_test, y_test), corrupt=True, sample_sizes=[200])
    plot_results(results_with_corruption, "K", y_label='Mean Error', group_by='k', title="k-NN Error vs. K With Corruption")
    print('Question 2.e Done')
