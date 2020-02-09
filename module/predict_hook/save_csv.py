from ..registry import HOOK
from utils.plot_utils import make_grid_img
import numpy as np
import pandas as pd


@HOOK.register("save_csv")
def save_csv(input_data=None, predict=None, target=None,
             save_path="/tmp"):
    '''
    func: save_csv
    args:
      save_path: plot.png
    '''
    num = len(predict)
    predict = np.argmax(predict, axis=1)

    result = pd.DataFrame({
        'ImageId': np.arange(1,num+1),
        'Label': predict
    })

    result.to_csv(save_path,index=0)

