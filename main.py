import sys
import numpy as np
from sklearn import model_selection
from preprocess import *

if __name__ == '__main__':
    np.random.seed(0)
    raw_data = pd.read_csv("waze_data.csv")
    train_data, test_data = model_selection.train_test_split(raw_data, test_size=.2, random_state=42)
    ls = preprocess(train_data)
    print("done")