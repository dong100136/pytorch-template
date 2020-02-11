from ..registry import HOOK
from utils.plot_utils import make_grid_img
import numpy as np
import pandas as pd
from pathlib import Path


@HOOK.register("save_csv")
def save_csv(dataset=None, 
             predict=None, 
             target=None,
             dim = -1,
             workspace = Path("/tmp"),
             save_path="submission.csv"):
    '''
    func: save_csv
    args:
      dim: dim<0 mean arg
      save_path: plot.png
    '''

    save_path = workspace / save_path
    num = len(predict)
    if dim>=0:
        predict = predict[:, dim]
    else:
        predict = np.argmax(predict, axis=1)

    imgs_path = dataset.imgs

    result = pd.DataFrame({
        'id': np.arange(1,num+1),
        'path': imgs_path,
        'label': predict
    })

    result.to_csv(save_path,index=0)

