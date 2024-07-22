"""
Example usage of the MulticlassSVM classifier on the Iris dataset.
"""
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from svm import MulticlassSVM
from svm.plots import plot_confusion_tables, plot_confusion_mat, \
    scatter_plot_mat

# number of classes in the Iris dataset
total_classes = 3

# dataset partitions (assuming they sum up to 1)
train_part = 0.5
val_part = 0.2
test_part = 0.3

# hyperparameter values for choosing between during validtion
C_list = [0.1]  # (soft-margin-SVM 'C' hyperparameter)
gamma_list = [0.25]  # (rbf kernel 'gamma' hyperparameter)


def main():
    # import the dataset and plot a scatter-plot-matrix of it
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    scatter_plot_mat(
        X, y,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        color_list=['r', 'g', 'b'],
        title='Iris Dataset:\nScatter Plot Matrix',
        alpha=0.3, scale=4
    )

    # split the dataset to training, validation, and test (we use a constant
    # RNG seed for reproducible splitting across multiple runs)
    train_val_X, test_X, train_val_y, test_y = train_test_split(
        X, y, test_size=test_part, random_state=0
    )
    train_X, val_X, train_y, val_y = train_test_split(
        train_val_X, train_val_y,
        test_size=val_part / (
                1 -
                test_part),
        random_state=0
    )

    # train MulticlassSVM with linear kernel to classify examples from the
    # Iris dataset, trying  out different C values and choosing the one with
    # the best performance on the validation set:
    best_C = -1
    best_acc = -1
    for C in C_list:
        print('Start fitting linear-kernel MulticlassSVM with C =', C,
              end=' ... ')
        curr_model = MulticlassSVM(kernel='linear', total_classes=total_classes,
                                   X=train_X, y=train_y, C=C)
        curr_model.fit()
        curr_acc = curr_model.calc_accuracy(val_X, val_y)
        print('Fitting complete. Validation accuracy =', curr_acc)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_C = C

    # fit a model with the best C value on the train & validation set
    print('Start fitting best linear-kernel MulticlassSVM with best_C =',
          best_C, end=' ... ')
    linear_model = MulticlassSVM(kernel='linear', total_classes=total_classes,
                                 X=train_val_X,
                                 y=train_val_y, C=best_C)
    linear_model.fit()
    print('Fitting complete')

    # analyze the resulting model - plot confusion matrix and confusion tables
    M = linear_model.calc_confusion_mat(test_X, test_y)
    plot_confusion_mat(M, class_names=iris.target_names,
                       title='Confusion Matrix\nIris dataset | Linear Kernel '
                             'MulticlassSVM | C = ' + str(
                           best_C))
    plot_confusion_tables(M, total_classes, iris.target_names,
                          title='Tables of Confusion\nIris dataset | Linear '
                                'Kernel MulticlassSVM | C = ' + str(
                              best_C))

    # train MulticlassSVM with rbf kernel to classify examples from the Iris
    # dataset, trying different C and gamma values and choosing the pair with
    # the best performance on the validation set:
    best_C = -1
    best_gamma = -1
    best_acc = -1
    for C in C_list:
        for gamma in gamma_list:
            print('Start fitting rbf-kernel MulticlassSVM with C =', C,
                  '| gamma =', gamma, end=' ... ')
            curr_model = MulticlassSVM(kernel='rbf',
                                       total_classes=total_classes, X=train_X,
                                       y=train_y, C=C, rbf_gamma=gamma)
            curr_model.fit()
            curr_acc = curr_model.calc_accuracy(val_X, val_y)
            print('Fitting complete. Validation accuracy =', curr_acc)
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_C = C
                best_gamma = gamma

    # fit a model with the best C and gamma values on the train & validation set
    print('Start fitting best rbf-kernel MulticlassSVM with best_C =', best_C,
          ' and best_gamma =', best_gamma,
          end=' ... ')
    gaussian_model = MulticlassSVM(kernel='rbf', total_classes=total_classes,
                                   X=train_val_X,
                                   y=train_val_y, C=best_C,
                                   rbf_gamma=best_gamma)
    gaussian_model.fit()
    print('Fitting complete')

    # analyze the resulting model - plot confusion matrix and confusion tables
    M = gaussian_model.calc_confusion_mat(test_X, test_y)
    plot_confusion_mat(M, class_names=iris.target_names,
                       title=(
                               'Confusion Matrix\nIris dataset | RBF '
                               'Kernel MulticlassSVM | C = ' +
                               str(best_C) + ' | gamma = ' + str(
                           best_gamma)))
    plot_confusion_tables(M, total_classes, iris.target_names,
                          title=(
                                  'Tables of Confusion\nIris dataset | '
                                  'RBF Kernel MulticlassSVM | C = ' +
                                  str(best_C) + ' | gamma = ' + str(
                              best_gamma)))

    plt.show()


if __name__ == '__main__':
    main()
