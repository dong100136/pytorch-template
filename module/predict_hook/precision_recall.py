import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report


def precision_recall(target, predict, save_path="/tmp"):
    '''
    predict : [N, 2] numpy array
    target : [N] numpy array
    '''
    predict = predict.argmax(axis=1)

    # p = accuracy_score(target, predict)
    # r = recall_score(target, predict)
    # f1 = f1_score(target, predict)

    # print("acc : %.4f\nrecall : %.4f\nf1 : %.4f"%(
    #     p,r,f1
    # ))

    print(classification_report(target,predict,digits=5))
