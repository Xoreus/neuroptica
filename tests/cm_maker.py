
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:42:03 2019

@author: simon
Edited: 04.10.2019

Creates a confusion matrix from general data
"""
from sklearn.metrics import confusion_matrix
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.utils.multiclass import unique_labels
import random


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
   # else:
        #print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    FOLDER = '../4x4_ONN/'
    gt_pred = np.array(random.choices(np.genfromtxt(FOLDER + 'gt_pred.csv', delimiter=','), k=50))

    zero = random.choices([(x, y) for (x,y) in gt_pred if x == 1], k=13)
    one = random.choices([(x, y) for (x,y) in gt_pred if x == 2], k=13)
    two = random.choices([(x, y) for (x,y) in gt_pred if x == 3], k=12)
    three = random.choices([(x, y) for (x,y) in gt_pred if x == 4], k=12)

    gt_pred = np.array(zero+one+two+three)

    gt = [int(x) for x in gt_pred[:,0]]
    pred = [int(x) for x in gt_pred[:,1]]


    #gt_50 = zero + one + two + three 

    #print(len(gt_50))


    #zero = [x for x in pred if x == 1]
    #one = [x for x in pred if x == 2]
    #two = [x for x in pred if x == 3]
    #three = [x for x in pred if x == 4]
    #
    ##pred_50 = random.choices(zero+one+two+three, k=50)
    #
    #pred_50 = random.choices(zero, k=13) + random.choices(one, k=13) + random.choices(two, k=12) + random.choices(three, k=12) 
    #
    #cm = confusion_matrix(gt, pred)
    ##print(sum(np.diag(cm)))
    #print(np.diag(cm))

    #ax = plot_confusion_matrix(gt_50, pred_50, [x for x in range(4)])
    ax = plot_confusion_matrix(gt, pred, [x for x in range(4)])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('Figures/cm_98.9%_50_samples_experimental_data.png')
    #ax2 = plot_confusion_matrix(cm, [x for x in range(4)], normalize=True)
    plt.show()
    #plt.savefig('Figures/normalized_cm_{}.pdf'.format(color))
