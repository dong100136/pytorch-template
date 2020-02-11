from ..registry import HOOK
from utils.plot_utils import make_grid_img
import numpy as np
import os

@HOOK.register("plot_result")
def plot_result(dataset=None, predict=None, target=None,
                ncols = 3, nrows=3,shuffle=True,
                label = None,
                label_map=None,
                workspace ="/tmp",
                save_path="plot.png"):
    '''
    func: plot_result
    args:
      save_path: plot.png
    '''
    save_path  = os.path.join(workspace,save_path)

    imgs_path = dataset.imgs
    prob = np.max(predict, axis=1)
    predict = np.argmax(predict,axis=1)

    if label_map:
        predict_label = [label_map[x] for x in predict]
    else:
        predict_label = [x for x in predict]

    make_grid_img(
        imgs_path, predict_label,prob = prob,
        nrows = nrows,ncols = ncols, shuffle=shuffle,
        save_path = save_path
    )
