"""
This file contains the SVM class - my implementation of a binary (soft-margin)
SVM classifier and its interface, fit using the SMO algorithm as introduced in:
"SequentiaL Minimal Optimization: A Fast Algorithm for Training Suppurt Vector
Machines", by J. C. Platt (1998).

Nadav Kahlon, December 2021
"""

import numpy as np
import random


class SVM:
    """
    A binary (soft-margin) SVM classifier.
    """

    def __init__(self, X, y, C=10, KKT_tol=0.001, signif_eps=0.001,
                 kernel='linear', rbf_gamma=None):
        """
        Creates a new classifier to be fitted on the given dataset.

        Note: smaller value for KKT_tol and signif_eps means allowing more
        subtle updates, achieving higher accuracy with the cost of longer fit
        time).

        :param X: an N by d real matrix whose rows are the N input vectors, of
            dimension d.
        :param y: array of N target values for classification (+1 or -1).
        :param C: the soft-margin-SVM hyperparameter - penalizing
            misclassifications.
        :param KKT_tol: tolerance for violation of the KKT conditions.
        :param signif_eps: small value, used to determine whether an update is
            not significant enough to be considered during the fitting process.
        :param kernel: the kernel being used. Supports: 'linear', and 'rbf';
            default='linear'.
        :param rbf_gamma: the gamma value used by rbf kernel (default
            calculated using the formula: 1 / (X.shape[1] * X.var())), if rbf
            kernel is chosen.
        """
        # setting up a few attributes:
        self.X = np.array(X)
        self.y = np.array(y)
        self.N = self.X.shape[0]  # size of training set
        self.C = C
        self.KKT_tol = KKT_tol
        self.signif_eps = signif_eps
        self.alphas = np.zeros(self.X.shape[
                                   0])  # array for the N lagrange
        # multipliers (initialized to 0)
        self.b = 0  # classification threshold

        # setting up the kernel function:
        if kernel == 'rbf':
            # setup gamma:
            if rbf_gamma == None:
                rbf_gamma = 1 / (self.X.shape[1] * self.X.var())
            self.kernel_func = lambda a, b: np.exp(
                -rbf_gamma * np.dot(a - b, a - b))
            self.is_linear = False
        else:  # default - linear kernel
            self.kernel_func = lambda a, b: np.dot(a, b)
        self.is_linear = (kernel != 'rbf')  # wether the kernel is linear or not
        self.w = np.zeros(
            self.X.shape[1])  # weight vector (used only with linear kernel)

        # setup the error cache: array of cached values of errors on training
        # examples (initialized to -y, since alphas are initialized to 0)
        self.E_cache = (-y).astype(np.float64)

    def evaluate(self, x):
        """
        Calculates the output of the SVM on a given input vector x.

        Note that this doesn't mean direct classification, but rather the output
        of the model only; to get actual classification you should run the sign
        function on that output.
        """
        if self.is_linear:
            # evaluation using linear kernel can be done directly with weight
            # vector
            return np.dot(self.w, x) - self.b
        else:
            # otherwise - evaluation is done using support vectors
            non_zero_examples = np.nonzero(self.alphas)[0]
            return sum(
                [self.y[n] * self.alphas[n] * self.kernel_func(self.X[n], x)
                 for n in non_zero_examples]) - self.b

    def calc_new_alphas(self, i1, i2):
        """
        Calculates the optimal lagrange multipliers for the i1'th and i2'th
        examples, optimizing over those multipliers only. Also calculates the
        new threshold b. Returns: (alpha1new, alpha2new, b_new). The returned
        values are None if the optimization has no (significant) effect.
        """
        # the implementation of this function is purely based on the algebra
        # explained in the above-mentioned paper.

        if i1 == i2: return None, None, None  # no optimization possible over
        # a single example
        # collect a few values necessary for the update rule
        x1 = self.X[i1]
        x2 = self.X[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.E_cache[i1]
        E2 = self.E_cache[i2]
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        s = y1 * y2
        if y1 != y2:
            L = max((0, alpha2 - alpha1))
            H = min((self.C, self.C + alpha2 - alpha1))
        else:
            L = max((0, alpha2 + alpha1 - self.C))
            H = min((self.C, alpha2 + alpha1))
        if (L == H): return None, None, None
        K11 = self.kernel_func(x1, x1)
        K12 = self.kernel_func(x1, x2)
        K22 = self.kernel_func(x2, x2)
        eta = K11 + K22 - 2 * K12

        # use eta, L, and H to determine the new value of alpha2
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if (a2 < L):
                a2 = L
            elif (a2 > H):
                a2 = H
        else:
            f1 = y1 * (E1 + self.b) - alpha1 * K11 - s * alpha2 * K12
            f2 = y2 * (E2 + self.b) - s * alpha1 * K12 - alpha2 * K22
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            L_obj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * K11 + 0.5 * (
                    L ** 2) * K22 + s * L * L1 * K12
            H_obj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * K11 + 0.5 * (
                    H ** 2) * K22 + s * H * H1 * K12
            if (L_obj < H_obj - self.signif_eps):
                a2 = L
            elif (L_obj > H_obj + self.signif_eps):
                a2 = H
            else:
                a2 = alpha2

        # make sure the update is significant enough
        if (abs(a2 - alpha2) < self.signif_eps * (
                a2 + alpha2 + self.signif_eps)): return None, None, None
        # and calculate the updated alpha1
        a1 = alpha1 + s * (alpha2 - a2)

        # calculate the new threshold b (as described in detail in the
        # above-mentioned paper):
        b1 = E1 + y1 * (a1 - alpha1) * K11 + y2 * (a2 - alpha2) * K12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * K12 + y2 * (a2 - alpha2) * K22 + self.b
        if (0 < a1 < self.C):
            b_new = b1
        elif (0 < a2 < self.C):
            b_new = b2
        else:
            b_new = (b1 + b2) / 2

        return a1, a2, b_new

    def update_E_cache(self, i1, i2, alpha1old, alpha2old, alpha1new, alpha2new,
                       b_old, b_new):
        """
        Updates the error cache according to 2 changes in the lagrange
        multipliers - the i1'th multiplier changed from 'alpha1old' to
        'alpha1new', and the i2'th multiplier changed from 'alpha2old' to
        'alpha2new'.
        """
        # collect a few values needed for the update
        x1 = self.X[i1]
        x2 = self.X[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]

        # simply scan every example, and update the error according to the
        # change in the evaluation function (follwing the changes in the alphas
        # and b)
        for I in range(self.N):
            x = self.X[I]
            y = self.y[I]

            # in case of linear kernel - error can be calculated directly
            # very fast
            if self.is_linear:
                self.E_cache[I] = self.evaluate(x) - y
            # otherwise - use the linearity of the output function
            else:
                self.E_cache[I] = (self.E_cache[I] + y1 * (
                        alpha1new - alpha1old) * self.kernel_func(x1, x) +
                                   y2 * (
                                           alpha2new - alpha2old) *
                                   self.kernel_func(
                                       x2, x) + b_old - b_new)

    def take_step(self, i1, i2):
        """
        Optimizes the classifier over the i1'th and i2'th examples.

        :return: Whether this optimization step had a significant-enough effect.
        """
        # collect a few values needed for the update
        x1 = self.X[i1]
        x2 = self.X[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]

        # record the old parameters (in order to update the error cache later)
        alpha1old = self.alphas[i1]
        alpha2old = self.alphas[i2]
        b_old = self.b

        # calculate the optimized lagrange multipliers and make sure the
        # optimization has a (significant) effect
        alpha1new, alpha2new, b_new = self.calc_new_alphas(i1, i2)
        if alpha1new == None or alpha2new == None: return False

        # store the new parameters in the model
        self.alphas[i1] = alpha1new
        self.alphas[i2] = alpha2new
        self.b = b_new

        if self.is_linear:
            # update weight vector (for linear kernel)
            self.w = self.w + y1 * (alpha1new - alpha1old) * x1 + y2 * (
                    alpha2new - alpha2old) * x2

        # update the error cache
        self.update_E_cache(i1, i2, alpha1old, alpha2old, alpha1new, alpha2new,
                            b_old, b_new)

        return True

    def examine_example(self, i2):
        """
        Optimizes the classifier over the i2'th example and another example
        (picked by the routine), if it does violate the KKT conditions;
        otherwise - does nothing. Returns 1 if an optimization occured, or 0 if
        the example does not violate the KKT conditions (or if no optimization
        is significant enough, so none took place).
        """

        # set a few values we are going to use
        y2 = self.y[i2]
        E2 = self.E_cache[i2]
        alpha2 = self.alphas[i2]
        r2 = E2 * y2
        # update, if the example violates the KKT conditions (up to some
        # tolerance):
        if (r2 < -self.KKT_tol and alpha2 < self.C) or (
                r2 > self.KKT_tol and alpha2 > 0):
            # use hierarchy of heuristics to pick the other example for
            # optimization, as explained in the above-mentioned paper:

            # first heuristic in the hierarchy:
            i1 = (np.argmin if E2 > 0 else np.argmax)(
                self.E_cache)  # pick an entry in the error cache according
            # to the first heuristic
            if self.take_step(i1, i2):
                return 1

            # second heuristic in the hierarchy:
            non_bounds = [I for I in range(self.N) if 0 < self.E_cache[
                I] < self.C]  # indices of non-bound examples
            random.shuffle(non_bounds)  # shuffle them
            for i1 in non_bounds:
                if self.take_step(i1, i2):
                    return 1

            # if no heuristic worked - simply try the rest of the dataset
            bounds = [I for I in range(self.N) if not (0 < self.E_cache[
                I] < self.C)]  # indices of bound examples
            random.shuffle(bounds)  # shuffle them
            for i1 in bounds:
                if self.take_step(i1, i2):
                    return 1

        return 0  # if no update occured - return 0

    def fit(self):
        """
        Uses the SMO algorithm to fit the classifier.
        """
        num_changed = 0  # counter for updates occurred during each iteration
        # of the outer loop
        examine_all = True  # whather the next iteration of the outer loop
        # should examine
        # the entire dataset, or only the non-bound subset

        # outer loop - examine the dataset until convergence, and update the
        # classifier
        while num_changed > 0 or examine_all:
            num_changed = 0  # reset update counter

            # examine the whole dataset:
            if examine_all:
                for I in range(self.N):
                    num_changed += self.examine_example(I)

            # examine the non-bound subset only (a heuristic used due to the
            # fact that
            # the non-bound subset more commonly violates the KKT conditions):
            else:
                for I in range(self.N):
                    E = self.E_cache[I]
                    if 0 < E < self.C:
                        num_changed += self.examine_example(I)

            # prepare for next iteration
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
