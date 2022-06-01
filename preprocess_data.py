import pandas as pd

from dates_coords import convert_dates, convert_coordinates


def categorize_linqmap_city(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=['linqmap_city'])


def preprocess_data(df: pd.DataFrame):
    convert_dates(df)
    convert_coordinates(df)
    categorize_linqmap_city(df)
    return df



def process_pubDate(df: pd.DataFrame):
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['pubDate'] = df['pubDate'].dt.year
    df['pubDate'] = df['pubDate'].dt.month
    df['pubDate'] = df['pubDate'].dt.week
    df['pubDate'] = df['pubDate'].dt.day
    df['pubDate'] = df['pubDate'].dt.hour
    df['pubDate'] = df['pubDate'].dt.minute
    df['pubDate'] = df['pubDate'].dt.dayofweek
    df = df.drop(["pubDate"], axis=1)
    return df


