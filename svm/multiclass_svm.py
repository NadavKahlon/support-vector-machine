"""
This file contains the MulticlassSVM class - my implementation of multiclass
classifier using the 1-vs-all strategy, based on SVMs as the basic binary
classifiers (fit using the SMO algorithm).

Nadav Kahlon, December 2021
"""

import numpy as np
from svm import SVM


class MulticlassSVM:
    """
    A 1-vs-all multiclass classifier, based on SVM binary classifiers (fit using
    the SMO algorithm).
    """

    def __init__(self, total_classes, X, y, C=10, KKT_tol=0.001,
                 signif_eps=0.001, kernel='linear', rbf_gamma=None):
        """
        Constructor: creates a new classifier to be fitted on the given dataset

        Note: smaller value for KKT_tol and signif_eps means allowing more
        subtle updates, achieving higher accuracy with the cost of longer fit
        time).

        :param total_classes: the number of classes for classification
        :param X: an N by d real matrix whose rows are the N input vectors,
            of dimension d.
        :param y: array of N target values for classification (0 through
            class_num-1 - one value for each class)
        :param C: the soft-margin-SVM hyperparameter - penalizing
            misclassifications (default=1).
        :param KKT_tol: tolerance for violation of the KKT conditions.
        :param signif_eps: small value, used to determine whether an update is
            not significant enough to be considered during the fitting process.
        :param kernel: the kernel being used. Supports: 'linear', and 'rbf'.
        :param rbf_gamma: the gamma value used by rbf kernel (default
            calculated using the formula: 1 / (X.shape[1] * X.var())), if rbf
            kernel is chosen.
        """
        self.total_classes = total_classes
        y = np.array(y)  # copy y to numpy array
        # create an array of 1-vs-all binary classifiers, one for each class:
        self.bin_SVMs = []
        for class_num in range(self.total_classes):
            curr_y = np.where(y == class_num, 1,
                              -1)  # target vector for current 1-vs-all
            # classifier
            self.bin_SVMs.append(
                SVM(X, curr_y, C, KKT_tol, signif_eps, kernel,
                    rbf_gamma))  # and add a corresponding SVM

    def fit(self):
        """
        Fits the multiclass classifier.
        """
        # simply fit every 1-vs-all binary classifier using SMO algorithm:
        for bin_SVM in self.bin_SVMs:
            bin_SVM.fit()

    def classify(self, x):
        """
        Predicts for which class the given d-dimensional input  vector 'x'
        belongs.
        """
        # we check which 1-vs-all binary classifier evaluates the highest
        # score for x
        scores = [bin_SVM.evaluate(x) for bin_SVM in self.bin_SVMs]
        return scores.index(max(scores))

    def calc_accuracy(self, X, y):
        """
        Calculates the accuracy of the classifier, based on the given
        validation
        set.

        :param X: matrix whose rows are validation examples.
        :param y: vector whose elements are the actual classes for which each
            example belongs.
        """
        predictions = [self.classify(example) for example in
                       X]  # predictions vector
        return np.count_nonzero(predictions == y) / len(
            X)  # percentage of correct predictions

    def calc_confusion_mat(self, X_test, y_test):
        """
        Calculates the confusion matrix of the classifier on the given test
        data.

        :param X_test: matrix whose rows are test examples.
        :param y_test: vector whose elements are the actual classes for which
            each example belongs.

        :return: square 'total_classes' by 'total_classes' confusion matrix;
            rows represent predicted classes and columns represent actual
            classes.
        """

        # create a list of pairs of predicted classes and actual classes for
        # the test examples:
        confusion_pairs = [[self.classify(X_test[i]), y_test[i]] for i in
                           range(len(X_test))]
        # build the confusion matrix based of the number of occurrences of
        # each pair:
        confusion_matrix = [[confusion_pairs.count([pred_class, act_class])
                             for act_class in range(self.total_classes)]
                            for pred_class in range(self.total_classes)]
        return np.array(confusion_matrix)
