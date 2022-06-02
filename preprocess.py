import typing
import pandas as pd
import numpy as np
import re
from geopy.distance import distance
from pyproj import Transformer

EMPTY = ['linqmap_reportDescription', 'linqmap_nearby',
         'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'OBJECTID',
         'nComments', 'linqmap_reportMood']

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

LOCATION_TIMEOUT = 6

EMPTY = ['linqmap_reportDescription', 'linqmap_nearby', 'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'OBJECTID', 'nComments', 'linqmap_reportMood']
def convert_dates(data: pd.DataFrame) -> None:
    dts = pd.to_datetime(data['update_date'], unit='ms')
    data['update_date'] = [dt.date() for dt in dts]
    data['update_time'] = [dt.time() for dt in dts]


def convert_coordinates(data) -> None:
    X = pd.Series.to_numpy(data['x'])
    Y = pd.Series.to_numpy(data['y'])

    # crs = CRS.from_epsg(6991)
    # crs.to_epsg()
    transformer = Transformer.from_crs("EPSG:6991", "EPSG:4326")
    wgs84_coords = [transformer.transform(X[i], Y[i]) for i in range(len(X))]
    data['y'] = [tup[0] for tup in wgs84_coords]
    data['x'] = [tup[1] for tup in wgs84_coords]


def categorize_linqmap_city(df: pd.DataFrame):
    pd.get_dummies(df, columns=['linqmap_city'])



def process_pubDate(df: pd.DataFrame):
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['pubDate'] = pd.to_datetime(df['pubDate'])  # full date "15/5/2022 20:30:55" as datetime
    df['date'] = pd.to_datetime(df['pubDate']).dt.date
    df['time'] = pd.to_datetime(df['pubDate']).dt.time
    df['pubDate_hour'] = df['pubDate'].dt.hour  # hour as int from 0 to 24
    df['pubDate_day_of_week'] = df['pubDate'].dt.dayofweek  # hour as int from 0 to 24


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


# def geolocator(coordinates: str) -> str:
#     """Return location coordinates and accurate address of the specified location."""
#     geolocator = Nominatim(user_agent="tutorial", timeout=LOCATION_TIMEOUT)
#     try:
#         location = geolocator.reverse(coordinates)
#         if location is not None:
#             return location.address
#     except GeocoderTimedOut as e:
#         print(str(e))
#     return ""

def get_nearest_location(x: float, y: float, df: pd.DataFrame) -> typing.Tuple[str, str]:
    x, y = float(x), float(y)
    min_dist = float('inf')
    city = ""
    street = ""
    for index, row in df.iterrows():
        x1, y1 = float(row['x']), float(row['y'])
        if x == x1 and y == y1:
            continue
        c1 = (y, x)
        c2 = (y1, x1)
        dist = distance(c1, c2)
        curr_city, curr_street = row['linqmap_city'], row['linqmap_street']
        if dist < min_dist and (curr_city or curr_street) and (
                curr_city is not np.nan or curr_street is not np.nan):
            city, street = curr_city, curr_street
            min_dist = dist
    return city, street


def add_accident_type(df: pd.DataFrame):
    # most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside of the city, put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    df['linqmap_subtype'].fillna(accident_type['linqmap_subtype'], inplace=True)

def proccess_road_closed(df):
    road_closed_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    null_road_closed_type = road_closed_type[road_closed_type['linqmap_subtype'].isna()]
    for idx, row in null_road_closed_type.iterrows():
        same_date = road_closed_type[road_closed_type["date"] == row['date']]
        if (not(row['linqmap_street'] is None)) and (same_date['linqmap_street'].str.contains(row['linqmap_street']).any()):
            df.at[idx,'linqmap_subtype'] = 'ROAD_CLOSED_CONSTRUCTION'
            continue
        row['linqmap_subtype'] = 'ROAD_CLOSED_EVENT'
    # print("hi")

def proccess_jam(df):
    jam_type = df[df["linqmap_type"] == "JAM"]
    null_jam_type = jam_type[jam_type['linqmap_subtype'].isna()]
    for idx, row in null_jam_type.iterrows():
        same_date_and_place = jam_type[(jam_type["date"] == row['date']) &
                                       (jam_type["linqmap_street"] == row['linqmap_street']) &
                                       (~(jam_type["linqmap_subtype"].isna()))]
        if same_date_and_place.empty:
            df.at[idx, 'linqmap_subtype'] = 'JAM_HEAVY_TRAFFIC'
            continue
        delta_time = same_date_and_place['pubDate'].apply(lambda x: np.abs((x - row['pubDate']).total_seconds()))
        closest = same_date_and_place.loc[delta_time.idxmin()]
        df.at[idx, 'linqmap_subtype'] = closest['linqmap_subtype']
        # print("h")

def proccess_weatherhazard(df):
    weatherhazard_type = df[df["linqmap_type"] == "WEATHERHAZARD"]
    dist = weatherhazard_type.linqmap_subtype.value_counts(normalize=True)
    missing = weatherhazard_type['linqmap_subtype'].isnull()
    weatherhazard_type.loc[missing, 'linqmap_subtype'] = np.random.choice(dist.index,
                                                 size=len(weatherhazard_type[missing]),
                                                 p=dist.values)
    df['linqmap_subtype'].fillna(weatherhazard_type['linqmap_subtype'], inplace=True)
    print('h')





def process_city_street(df: pd.DataFrame):
    n = 0
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
        print(n)
        # city and street is missing (we search for the nearset coordinates and fill the missing data)
        # if df['linqmap_city'][index] == 'Out of district' and df['linqmap_street'][index] == 0:
        #     print("searching....")
        #     nearest_city, nearest_street = get_nearest_location(df['x'][index], df['y'][index], df)
        #     df['linqmap_city'][index] = nearest_city
        #     df['linqmap_street'][index] = nearest_city

        n+=1


def preprocess(df: pd.DataFrame) -> None:
    process_pubDate(df)
    convert_coordinates(df)
    convert_dates(df)
    proccess_accident(df)
    proccess_road_closed(df)
    proccess_jam(df)
    proccess_weatherhazard(df)
    categorize_linqmap_city(df)
    remove_diluted_features(df)
    process_city_street(df)
