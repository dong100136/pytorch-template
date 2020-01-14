import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns


def confusion_matrix(target, predict, save_path="/tmp"):
    '''
    predict : [N, 2] numpy array
    target : [N] numpy array
    '''


    plt.figure(dpi=200)
    confusion = metrics.confusion_matrix(target, predict.argmax(axis=1))

    print("confusion matrix:")
    print(confusion)

    
    sns.heatmap(confusion, annot=True, cmap="BuPu")

    plt.savefig(save_path / "confusion.png")
    print("save confusion matrix at %s" % (save_path / "confusion.png"))

    