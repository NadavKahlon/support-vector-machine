"""
This file includes a few utility functions for analyzing and plotting the data.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def plot_confusion_mat(M, class_names, title=None):
    """
    Plots the confusion matrix 'M' of a multi-class classifier on whole new
    window. Adds 'title' to the plot (if not None). Also adds information
    about the classifier's accuracy to the title (calculated using confusion
    matrix). Assumes 'class_names' is a list containing names for the different
    classes.
    """

    fig, axis = plt.subplots()
    plt.subplots_adjust(top=0.770)
    # show the matrix and add a colorbar
    img_plot = plt.imshow(M, cmap='YlOrBr')
    plt.colorbar(img_plot, ax=axis)
    # print its values on it
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axis.text(j, i, M[i][j], color='deepskyblue', fontsize='xx-large',
                      weight='bold' if i == j else 'medium', va='center',
                      ha='center')
    # setup x and y axes
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)
    plt.setp(axis, ylabel='Predicted Class', xlabel='Actual Class')
    axis.xaxis.set_label_position('top')
    axis.xaxis.tick_top()
    # setup title
    title = '' if title == None else (title + '\n')
    title += 'Accuracy: '
    title += str(
        round(100 * M.trace() / M.sum(), 2)) + '%'  # accuracy evaluation
    axis.set_title(title)


def plot_confusion_table(ax, M, class_num, title=None):
    """
    lots the table of confusion for the 'class_num' class on the subplot axis
    'ax', calculated using a classifier's confusion matrix 'M'. Adds 'title' to
    the plot (if not None). Also adds information about the classifier's
    sensitivity for that class.
    """

    # create a copy of M, and switch between rows and columns to make
    # 'class_num' the
    # first class in the matrix (for easier computation)
    M = np.array(M)
    M[[0, class_num], :] = M[[class_num, 0], :]
    M[:, [0, class_num]] = M[:, [class_num, 0]]
    # calculate true/false positive/negative
    TP = M[0, 0]
    FP = M[0, 1:].sum()
    FN = M[1:, 0].sum()
    TN = M[1:, 1:].sum()
    # plot the data and add a colorbar
    ax.imshow([[TP, FP],
               [FN, TN]], cmap='GnBu')
    # print the data on the plot
    ax.text(0, 0, 'TP=' + str(TP), color='r', fontsize='small', weight='bold',
            va='center', ha='center')
    ax.text(1, 0, 'FP=' + str(FP), color='r', fontsize='small', weight='medium',
            va='center', ha='center')
    ax.text(0, 1, 'FN=' + str(FN), color='r', fontsize='small', weight='medium',
            va='center', ha='center')
    ax.text(1, 1, 'TN=' + str(TN), color='r', fontsize='small', weight='bold',
            va='center', ha='center')
    # setup x and y axes
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])
    # setup title
    title = '' if title == None else (title + '\n')
    title += 'Sensitivity: '
    title += str(round(100 * TP / (TP + FN), 2)) + '%'  # sensitivity evaluation
    ax.set_title(title)


def plot_confusion_tables(M, total_classes, class_names, title):
    """
    Plots tables of confusion and the sensitivity of a classifier for all the
    'total_classes' classes, on a whole new window; calculations are done
    using the given table of cunfusion 'M'. Adds 'title' to the plot (if not
    None). Assumes 'class_names' is a list of the classes' names.
    """
    # create subplot for each table of confusion:
    fig, axes = plt.subplots(1, total_classes)
    for i in range(total_classes):
        plot_confusion_table(axes[i], M, class_num=i,
                             title=('Class: ' + class_names[i]))
    # setup the title:
    fig.suptitle(title)


def scatter_plot_mat(X, y, feature_names, class_names, color_list, title=None,
                     alpha=0.3, scale=4):
    """
    Plots a scatter plot matrix for a classes-based dataset (on a whole window).

    :param X: an N by d real matrix whose rows are the N vectors of dimension d.
    :param y: array of N values representing classes of the input vectors (from
        0 to the number of classes minus 1).
    :param feature_names: list of feature names.
    :param class_names: list of the classes' names.
    :param color_list: list of colors to draw each class' points with
    :param title: optional title (actually displayed in the legend)
    :param alpha: transparency of the points (default=0.3).
    :param scale: scale of the points (default=4)
    """

    d = X.shape[1]  # number of features
    # create matrix of subplots:
    fig, axes = plt.subplots(nrows=d, ncols=d, sharex='col', sharey='row',
                             figsize=(7.5, 6))
    for i in range(d):
        for j in range(d):
            # setup the subplot:
            if i == j:
                # subplot on the diagonal - we write the feature's name
                axes[i][j].text(0.5, 0.5, feature_names[i], ha='center',
                                va='center',
                                size=9, weight='bold',
                                transform=axes[i][j].transAxes)
            else:
                axes[i][j].scatter(X[:, j],  # features on the horizontal axis
                                   X[:, i],  # features on the vertical axis
                                   s=scale, alpha=alpha,
                                   c=[color_list[class_i] for class_i in
                                      y])  # color each point by its class
    # setup legend (one element for each class) and other cosmetics:
    legend_elements = [
        Line2D([0], [0], label=class_names[k], color=color_list[k], marker='o')
        for k in range(len(class_names))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), title=title,
               loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
