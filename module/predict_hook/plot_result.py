from ..registry import HOOK
from utils.plot_utils import make_grid_img
import numpy as np

@HOOK.register("plot_result")
def plot_result(input_data=None, predict=None, target=None,
                ncols = 3, nrows=3,shuffle=True,
                save_path="/tmp"):
    '''
    func: plot_result
    args:
      save_path: plot.png
    '''
    input_data = input_data.transpose(0,2,3,1)
    input_data = np.squeeze(input_data)
    predict = np.argmax(predict,axis=1)

    make_grid_img(
        input_data, predict,
        nrows = nrows,ncols = ncols, shuffle=shuffle,
        save_path = save_path
    )
