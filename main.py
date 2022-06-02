import sys
import warnings

from sklearn.model_selection import train_test_split
import pandas as pd
from preprocess import *

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    try:
        geolocator_use = sys.argv[1]
    except IndexError:
        geolocator_use = ''
    np.random.seed(0)
    raw_data = pd.read_csv("waze_data.csv")
    train_data, test_data = train_test_split(raw_data, test_size=.2, random_state=42)
    data = preprocess(train_data, True if geolocator_use == "-g" else False)
    print("done")