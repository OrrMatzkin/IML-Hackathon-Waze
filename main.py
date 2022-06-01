import sys

import numpy as np

from load_data import load_data

if __name__ == '__main__':
    print("GOOD LUCK")
    np.random.seed(0)
    X_train, X_test = load_data(f"waze_data.csv")