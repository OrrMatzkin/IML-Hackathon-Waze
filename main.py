from os.path import exists
from task_1 import run_task_1
from task_2 import run_task_2
import pandas as pd
import numpy as np
import warnings
import sys


if __name__ == '__main__':
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    try:
        train_data = sys.argv[1]
        test_data_task_1 = sys.argv[2]
    except IndexError:
        sys.stderr.write("Error: missing variables, Usage: <train_data> <test_data_task_1>")
    else:
        if not exists(train_data):
            sys.stderr.write("Error: train data file does not exist!")
        elif not exists(test_data_task_1):
            sys.stderr.write("Error: test data for task 1 file does not exist!")
        else:
            # runs both tasks
            raw_train_data = pd.read_csv(train_data)
            df1 = run_task_1(raw_train_data, pd.read_csv(test_data_task_1))
            raw_train_data = pd.read_csv(train_data)
            df2 = run_task_2(raw_train_data)
            # saves predictions to csv
            with open("predictions/task1_predictions.csv", "w+") as my_csv:
                df1.to_csv(my_csv, index=False)

            for i in range(len(df2)):
                with open(f"predictions/task2_predictions{i+1}.csv", "w+") as my_csv:
                    df = pd.DataFrame.from_records(df2[i], columns=["ACCIDENT", "JAM",
                                                                    "ROAD_CLOSED",
                                                                    "WEATHERHAZARD"])
                    df.to_csv(my_csv, index=False)
