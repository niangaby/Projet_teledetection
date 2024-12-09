#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confusion Matrix Visualization Tool

This module provides functionality to create detailed confusion matrix visualizations
with precision, recall, and F1-score metrics.

Created on Jan 06 2020
Last modified: Oct 27 2024

@author: Yousra Hamrouni
@organization: Toulouse INP, Dynafor 
"""


from sklearn.metrics import confusion_matrix
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from matplotlib import rcParams

# Some useful mpl personalizations for plots rendering 
# Comment if you want to use the default matplotlib backend
rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'axes.labelsize': 'large',
    'axes.labelweight': 'bold',
    'axes.titlesize': 'large',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def plot_confusion_matrix(y_test, y_test_pred, 
                 class_names=None,
                 outdir=os.getcwd(),
                 normalize=False, 
                 cmap='Greens'):
    """
    Plot a confusion matrix with precision, recall, and F1-score metrics.
    
    Parameters:
    ----------
    cm : np.array
        Confusion matrix, reference are expected in rows and prediction in
        columns
    class_names : list of str, optional
        Names of the classes. If None, uses the unique values from y_test_pred.
    outdir : str, optional
        Directory to save the output plot (default is the current working directory).
    normalize : bool, optional
        If True, normalize the confusion matrix by converting counts to percentages (default is False).
    cmap : str, optional
        Colormap to use for the confusion matrix visualization (default is 'Greens').
    
    The function generates and saves a confusion matrix plot with precision, recall, 
    and F1-score metrics displayed alongside. The confusion matrix can be normalized 
    for better interpretation.
    """
    
    # Sort labels for consistent ordering
    labels = sorted(np.unique(y_test_pred))
    confmat = confusion_matrix(y_test, y_test_pred)
    
    # Use labels as class names if not provided
    if class_names is None:
        class_names = [str(label) for label in labels]  
    
    # Calculate precision, recall, and F1 score
    precision = confmat.diagonal() / confmat.sum(axis=0) * 100
    recall = confmat.diagonal() / confmat.sum(axis=1) * 100
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Class names and values
    class_values = list(range(len(class_names)))
    dic = dict(zip(class_values, class_names))
    
    # Normalization and limits
    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis] * 100
        max_value = 100
        name = "confmat_normalized"
    else:
        max_value = confmat.max()
        name = "confmat"
    
    # Setup figure and gridspec
    n_classes = len(labels)
    # fig_width = 3 * n_classes  # Width of the figure
    # fig_height = fig_width / 2  # Fixed height, can be adjusted as needed
    fig = plt.figure(figsize=(15, 6))
    
    # Create gridspec for subplots
    gs = fig.add_gridspec(1, 3, width_ratios=[n_classes, 3, 0.2], wspace=0.2)    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax_percent = fig.add_subplot(gs[0, 2])
    
    # Plot confusion matrix
    cax1 = ax1.matshow(confmat, vmin=0, vmax=max_value, cmap=cmap, alpha=0.75)
    
    # Plot precision and recall metrics
    metrics = np.zeros((confmat.shape[0], 3))
    metrics[:, 0] = precision
    metrics[:, 1] = recall
    metrics[:, 2] = f1_score
    
    ax2.set_aspect(2 / n_classes)
    cax2 = ax2.matshow(metrics, vmin=0, vmax=100, cmap=cmap, alpha=0.75)
    
    # Colorbars and labels
    if normalize:
        cbar_percent = fig.colorbar(cax2, cax=cbar_ax_percent, orientation='vertical')
        cbar_percent.set_label('Score (\%)', rotation=90, labelpad=5)
        cbar_percent.set_ticks(np.linspace(0, 100, 5))
    else:
        cbar_ax_count = cbar_ax_percent.twinx()
        cbar_percent = fig.colorbar(cax2, cax=cbar_ax_percent, orientation='vertical')
        cbar_percent.set_ticks(np.linspace(0, 100, 5))
        
        # Colorbar for counts scale (on the left)
        cbar_count = fig.colorbar(cax1, cax=cbar_ax_count, orientation='vertical')
        
        fig.canvas.draw() 
        cbar_width = cbar_percent.ax.get_window_extent().width

        # Dynamically adjust labelpad based on the colorbar width
        cbar_position = cbar_percent.ax.get_position()
        cbar_count.set_label('Pixel count', rotation=90, labelpad=-30 - cbar_position.x0 - cbar_width)
        
        cbar_percent.set_label('Score (\%)', rotation=90, labelpad=30 + cbar_position.x1 - cbar_width)

        cbar_count.set_ticks(np.arange(0, max_value + 1, max_value // 4))  # Ensure max_value is an integer
        cbar_ax_count.yaxis.set_ticks_position('left')

    # Confusion matrix labels and values
    tick_marks = np.arange(len(labels))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels([dic[x] for x in labels], rotation=50)
    ax1.set_yticklabels([dic[x] for x in labels])
    
    fmt = '.1f' if normalize else 'd'
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        ax1.text(j, i, f"{confmat[i, j]:{fmt}}",
                 horizontalalignment="center", color="black")

    # Precision-Recall labels and values
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Precision', 'Recall', 'F1_score'], rotation=50)
    ax2.set_yticks([])
    
    for i in range(len(labels)):
        ax2.text(0, i, f'{precision[i]:.1f}', horizontalalignment="center", color="black")
        ax2.text(1, i, f'{recall[i]:.1f}', horizontalalignment="center", color="black")
        ax2.text(2, i, f'{f1_score[i]:.1f}', horizontalalignment="center", color="black")

    ax1.set_ylabel('True labels', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_xlabel('Predicted labels', fontweight='bold', fontsize=14, labelpad=10)

    # Customize tick parameters
    for ax in [ax1, ax2]:
        ax.tick_params(bottom=False, right=False)
        ax.tick_params(left=True, top=True, length=5)
        ax.xaxis.tick_top()
    
    # Save the figure
    plt.savefig(f"{outdir}/{name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()


# Testing part: create toy 2D data and fit the model
if __name__ == "__main__":
    # Create toy 2D data
    X, y = make_blobs(n_samples=100,  
                      centers=5,  # 5 classes
                      n_features=2,  # 2 input variables
                      random_state=0)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Fit a logistic regression classifier
    model = LR(random_state=0, C=1, max_iter=1000)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    # Call the plot function
    plot_confusion_matrix(y_test, y_test_pred, 
                 normalize=True,  # optionnal normalize the confusion matrix
                 class_names=class_names)  # optionnal personnalize class names
