import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def FROC(target, predict, save_path="/tmp"):
    '''
    predict : [N, 2] numpy array
    target : [N] numpy array
    '''
    fpr, tpr, thr = roc_curve(target, predict[:,1])
    auc_score = auc(fpr, tpr)

    plt.figure(dpi=200)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(save_path / "roc_curve.png")
    print("save roc curve at %s" % (save_path / "roc_curve.png"))
