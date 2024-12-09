# -*- coding: utf-8 -*-
"""
Created on 06/4/2020

@author: Marc LANG, Yousra HAMROUNI
@mail: marc.lang@toulouse-inp.fr
"""
from matplotlib.pyplot import cm as colorMap
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import itertools

def custom_bg(ax, x_label=None, y_label=None, fontsize=18, labelsize=14,
              x_grid=True, y_grid=True, minor=True):

    ax.set_facecolor('ivory')

    # custom label
    x_label = ax.get_xlabel() if not x_label else x_label
    ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=fontsize)
    y_label = ax.get_ylabel() if not y_label else y_label
    ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=fontsize)

    # custom border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='x', colors='darkslategrey', labelsize=labelsize)
    ax.tick_params(axis='y', colors='darkslategrey', labelsize=labelsize)

    # custom grid
    if minor:
        ax.minorticks_on()
    if y_grid:
        ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                      linewidth=0.5, zorder=1)
        ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                      linewidth=0.3, zorder=1)
    if x_grid:
        ax.xaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                      linewidth=0.5, zorder=1)

        ax.xaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                      linewidth=0.3, zorder=1)
    return ax


def plot_class_quality(report, accuracy, out_filename=None):
    """
    Display a plot bar of quality metrics of each class.

    Parameters
    ----------
    report : dict
        Classification report (output of the `classification_report` function
        of scikit-learn.
    accuracy : float
        Overall accuracy.
    out_filename : str (optional)
        If indicated, the chart is saved at the `out_filename` location
    """
    report_df = pd.DataFrame.from_dict(report)
    # drop columns (axis=1) same as numpy
    try :
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'],
                                   axis=1)
    except KeyError:
        report_df = report_df.drop(['micro avg', 'macro avg', 'weighted avg'],
                                   axis=1)
    # drop rows (axis=0) same as numpy
    report_df = report_df.drop(['support'], axis=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = report_df.T.plot.bar(ax=ax, zorder=2)

    # custom : information
    ax.text(0.05, 0.95, 'OA : {:.2f}'.format(accuracy), fontsize=14)
    ax.set_title('Class quality estimation')

    # custom : cuteness
    # background color
    ax.set_facecolor('ivory')
    # labels
    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    y_label = ax.get_ylabel()
    ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    # borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
    ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
    # grid
    ax.minorticks_on()
    ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                  linewidth=0.5, zorder=1)
    ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                  linewidth=0.3, zorder=1)
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight')


def plot_mean_class_quality(list_df_report, list_accuracy, out_filename=None):
    """
    Display a plot bar of quality metrics of each class.

    Parameters
    ----------
    report : dict
        Classification report (output of the `classification_report` function
        of scikit-learn.
    accuracy : float
        Overall accuracy.
    out_filename : str (optional)
        If indicated, the chart is saved at the `out_filename` location
    """

    # compute mean of accuracy
    array_accuracy = np.asarray(list_accuracy)
    mean_accuracy = array_accuracy.mean()
    std_accuracy = array_accuracy.std()

    array_report = np.array(list_df_report)
    mean_report = array_report.mean(axis=0)
    std_report = array_report.std(axis=0)
    a_report = list_df_report[0]
    mean_df_report = pd.DataFrame(mean_report, index=a_report.index,
                                  columns=a_report.columns)
    std_df_report = pd.DataFrame(std_report, index=a_report.index,
                                 columns=a_report.columns)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax = mean_df_report.T.plot.bar(ax=ax,
                                   yerr=std_df_report.T, zorder=2)
    # custom : information
    ax.set_ylim(0.5, 1)
    ax.text(1.5, 0.95, 'OA : {:.2f} +- {:.2f}'.format(mean_accuracy,
                                                      std_accuracy),
            fontsize=14)
    ax.set_title('Class quality estimation')

    # custom : cuteness
    # background color
    ax.set_facecolor('ivory')
    # labels
    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    y_label = ax.get_ylabel()
    ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    # borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
    ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
    # grid
    ax.minorticks_on()
    ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                  linewidth=0.5, zorder=1)
    ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                  linewidth=0.3, zorder=1)
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight')
        

def plot_cm(cm, labels, out_filename=None,
            normalize=False,  cmap='Greens'):
    """
    Plot a confusion matrix with precision, recall, and F1-score metrics.
    
    Parameters:
    ----------
    cm : np.array
        Confusion matrix, reference are expected in rows and prediction in
        columns
    labels : list of str
        Names of the classes.
    out_filename : str (optional)
        If indicated, the chart is saved at the `out_filename` location
    normalize : bool, optional
        If True, normalize the confusion matrix by converting counts to percentages (default is False).
    cmap : str, optional
        Colormap to use for the confusion matrix visualization (default is 'Greens').
    
    The function generates and saves a confusion matrix plot with precision, recall, 
    and F1-score metrics displayed alongside. The confusion matrix can be normalized 
    for better interpretation.
    """
    
    
    # Calculate precision, recall, and F1 score
    precision = cm.diagonal() / cm.sum(axis=0) * 100
    recall = cm.diagonal() / cm.sum(axis=1) * 100
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Class names and values
    class_values = list(range(len(labels)))
    dic = dict(zip(class_values, labels))
    
    # Normalization and limits
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        max_value = 100
        name = "cm_normalized"
    else:
        max_value = cm.max()
        name = "cm"
    
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
    cax1 = ax1.matshow(cm, vmin=0, vmax=max_value, cmap=cmap, alpha=0.75)
    
    # Plot precision and recall metrics
    metrics = np.zeros((cm.shape[0], 3))
    metrics[:, 0] = precision
    metrics[:, 1] = recall
    metrics[:, 2] = f1_score
    
    ax2.set_aspect(2 / n_classes)
    cax2 = ax2.matshow(metrics, vmin=0, vmax=100, cmap=cmap, alpha=0.75)
    
    # Colorbars and labels
    if normalize:
        cbar_percent = fig.colorbar(cax2, cax=cbar_ax_percent, orientation='vertical')
        cbar_percent.set_label('Score (%)', rotation=90, labelpad=5)
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
        
        cbar_percent.set_label('Score (%)', rotation=90, labelpad=30 + cbar_position.x1 - cbar_width)

        cbar_count.set_ticks(np.arange(0, max_value + 1, max_value // 4))  # Ensure max_value is an integer
        cbar_ax_count.yaxis.set_ticks_position('left')

    # Confusion matrix labels and values
    tick_marks = np.arange(n_classes)
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(labels, rotation=50)
    ax1.set_yticklabels(labels)
    
    fmt = '.1f' if normalize or isinstance(cm[0,0], float) else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax1.text(j, i, f"{cm[i, j]:{fmt}}",
                 horizontalalignment="center", color="black")

    # Precision-Recall labels and values
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Precision', 'Recall', 'F1_score'], rotation=50)
    ax2.set_yticks([])
    
    for i in range(n_classes):
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
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight', dpi=300)
