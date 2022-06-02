import math
from datetime import datetime
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
import re

EMPTY = ['linqmap_reportDescription', 'linqmap_nearby', 'linqmap_expectedBeginDate', 'linqmap_expectedEndDate',
         'OBJECTID', 'nComments', 'linqmap_reportMood']

DISTRICTS_OF_ISRAEL = {"North District": ['בית שאן', 'טבריה', 'טמרה', 'יקנעם עלית', 'כרמיאל', 'מגדל העמק',
                                          "מע'אר", 'מעלות תרשיחא', 'נהריה', 'נוף הגליל', 'נצרת', "סחנין"
                                          'עראבה', 'עכו', 'עפולה', 'צפת', 'קריית שמונה', 'שפרעם'],
                       "Haifa District": ['אום אל - פאחם', 'אור עקיבא', 'באקה אל גרביה', 'חדרה', 'חיפה', 'טירת כרמל',
                                          'נשר', 'קריית אתא', 'קריית ביאליק', 'קריית ים', "קריית מוצקין", 'קריית'],
                       "Tel Aviv District": ['אור יהודה', 'בני ברק', 'בת ים', 'גבעתיים', 'הרצליה', 'חולון',
                                             'אונו קריית', 'רמת גן', 'רמת השרון', 'תל אביב - יפו', "קריית מוצקין"],
                       "Center District": ['אלעד', 'באר יעקב', 'גבעת שמואל', 'הוד השרון', 'טייבה', 'יבנה',
                                           "יהוד-מונוסון", 'כפר יונה', 'כפר סבא', 'כפר קאסם', 'לוד', "מודיעין",
                                           'נס ציונה', 'נתניה', 'פתח תקווה', 'קלנסווה', 'רעש העין', 'ראשון לציון',
                                           'רחובות', 'רמלה', 'רעננה'],
                       "Jerusalem District": ['בית שמש', 'ירושלים'],
                       "Southern District": ['אופקים', 'אילת', 'אשדוד', 'אשקלון', 'באר שבע', 'דימונה',
                                             'נתיבות', 'ערד', 'קריית גת', 'קריית מלאכי', "רהט", 'שדרות'],
                       "Judea and Samaria District": ['אריאל', 'ביתר עילית', 'מודיעין עילית', 'מעלה אדומים', 'מודיעין']}



def convert_dates(data: pd.DataFrame) -> None:
    dates = data['update_date']
    data['update_date'] = pd.Series([datetime.fromtimestamp(time / 1000) for time in dates])


def convert_coordinates(data) -> None:
    X = np.array(data['x'])
    Y = np.array(data['y'])

    crs = CRS.from_epsg(6991)
    crs.to_epsg()
    crs = CRS.from_proj4(
        "+proj=tmerc +lat_0=31.7343936111111 +lon_0=35.2045169444445 "
        "+k=1.0000067 +x_0=219529.584 +y_0=626907.39 +ellps=GRS80 "
        "+towgs84=-24.002400,-17.103200,-17.844400,-0.33007,-1.852690,"
        "1.669690,5.424800 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:6991", "EPSG:4326")
    wgs84_coords = [transformer.transform(X[i], Y[i]) for i in range(len(X))]
    data['y'] = pd.Series([tup[0] for tup in wgs84_coords])
    data['x'] = pd.Series([tup[1] for tup in wgs84_coords])


def categorize_linqmap_city(df: pd.DataFrame):
    pd.get_dummies(df, columns=['linqmap_city'])


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


def remove_diluted_features(df: pd.DataFrame, diluted_proportion: float = .9) -> list:
    df.drop_duplicates(subset=['OBJECTID'], inplace=True)
    features = []
    n_samples = df.shape[0]
    for feature in df:
        num_empty_cell = df[feature].isnull().sum()
        if num_empty_cell / n_samples >= diluted_proportion:
            features.append(feature)
    features += ['OBJECTID', 'nComments']
    df.drop(EMPTY, axis=1, inplace=True)
    return features


def add_accident_type(df: pd.DataFrame):
    # most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside of the city, put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    # ROAD_CLOSED_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    # fig = px.scatter(ROAD_CLOSED_type, x="linqmap_street", y="linqmap_subtype")
    # fig.show()


def process_city_street(df: pd.DataFrame):
    df['linqmap_street'].fillna(0, inplace=True)
    for index, sample in df.iterrows():
        curr_city = sample['linqmap_city']
        curr_street = sample['linqmap_street']
        found_district = False
        # update city district
        for district, cities in DISTRICTS_OF_ISRAEL.items():
            if curr_city in cities:
                df['linqmap_city'][index] = district
                found_district = True
        if not found_district:
                df['linqmap_city'][index] = 'Out of district'

        # update street
        if curr_street != 0:
            road_numbers = re.findall("[0-9]+", curr_street)
            if len(road_numbers) > 0:
                df['linqmap_street'][index] = int(road_numbers[0])
            else:
                iter = re.finditer('ל-', curr_street)
                indices = [m.start(0) for m in iter]
                if len(indices) > 0:
                    curr_street = curr_street[::-1]
                    curr_street = curr_street[:len(curr_street) - (indices[0] + 2)].strip()[::-1]
                    df['linqmap_street'][index] = curr_street

        # city and street is missing (we search for the nearset coordinates and fill the missing data)
        if df['linqmap_city'][index] == 'Out of district' and df['linqmap_street'][index] == 0:
            nearest_city, nearest_street = get_nearest_location(df['x'][index], df['y'][index], df)
            df['linqmap_city'][index] = nearest_city
            df['linqmap_street'][index] = nearest_city


def preprocess(df: pd.DataFrame) -> None:
    add_accident_type(df)
    convert_dates(df)
    convert_coordinates(df)
    categorize_linqmap_city(df)
    remove_diluted_features(df)
    process_city_street(df)
