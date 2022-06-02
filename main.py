from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import *

if __name__ == '__main__':
    np.random.seed(0)
    raw_data = pd.read_csv("waze_data.csv")
    train_data, test_data = train_test_split(raw_data, test_size=.2, random_state=42)
    preprocess(train_data)
    print("done")