from pathlib import Path
import numpy as np
from scipy.io import loadmat

data_dir = Path(__file__).parents[2].joinpath('data')


def load_data(data_path='data_for_pda.mat'):
    loaded_data: dict = loadmat(str(data_dir.joinpath(data_path)))
    K = loaded_data["K"].item()
    Ts = loaded_data["Ts"].item()
    Xgt = loaded_data["Xgt"].T
    Z = [zk.T for zk in loaded_data["Z"].ravel()]
    true_association = loaded_data["a"].ravel()
    return K, Ts, Xgt, Z, true_association
