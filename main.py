from os.path import exists
from task_1 import run_task_1
from task_2 import run_task_2
from sklearn.metrics import f1_score, mean_squared_error
import sys
import warnings
import plotly.graph_objects as go
from preprocess import *
from fit_kit import *


def generate_pearson_correlation_heatmap(df):
    corr_df = df.corr()
    go.Figure([go.Heatmap(x=df.columns, y=df.columns, z=corr_df, type='heatmap', colorscale='Viridis')]).show(renderer="browser")


def evaluate_location(y_x_predict: pd.DataFrame, y_x_true: pd.DataFrame,
                     y_y_predict: pd.DataFrame, y_y_true: pd.DataFrame) -> float:
    return (mean_squared_error(y_x_predict, y_x_true) + mean_squared_error(y_y_predict, y_y_true))/2


def evaluate_type(y_type_predict: pd.DataFrame, y_type_true: pd.DataFrame) -> float:
    return f1_score(y_type_true, y_type_predict, average="macro")


if __name__ == '__main__':
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    try:
        train_data = sys.argv[1]
        test_data_task_1 = sys.argv[2]
        test_data_task_2 = sys.argv[3]
    except IndexError:
        sys.stderr.write("Error: missing variables, Usage: <train_data> <test_data_task_1> <test_data_task_1>")
    else:
        if not exists(train_data):
            sys.stderr.write("Error: train data file does not exist!")
        elif not exists(test_data_task_1):
            sys.stderr.write("Error: test data for task 1 file does not exist!")
        elif not exists(test_data_task_2):
            sys.stderr.write("Error: test data for task 2 file does not exist!")
        else:
            # runs both tasks
            raw_train_data = pd.read_csv(train_data)
            run_task_1(raw_train_data, pd.read_csv(test_data_task_1))
            run_task_2(raw_train_data, pd.read_csv(test_data_task_1))
