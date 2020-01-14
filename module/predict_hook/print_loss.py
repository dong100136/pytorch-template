import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from pathlib import Path
import pickle


def print_loss(target, predict, save_path=Path("/tmp")):
    '''
    predict : [N, 2] numpy array
    target : [N] numpy array
    '''
    loss = -(1-target)*np.log(predict[:,0])-target*np.log(predict[:,1])

    with open(save_path/"predict.pkl",'wb') as f:
        pickle.dump({
            'loss': loss,
            'predict': predict,
            'target': target
        },f)
