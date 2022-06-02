import typing
import pandas as pd
import numpy as np
import re
from geopy import Nominatim
from geopy.distance import distance
from geopy.exc import GeocoderTimedOut
from pyproj import Transformer

FEATURES_TO_DROP = ['linqmap_reportDescription', 'linqmap_nearby',
                    'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'OBJECTID', 'nComments',
                    'linqmap_reportMood', 'linqmap_magvar', 'update_date_new', 'update_date',
                    'update_time', 'pubDate', 'linqmap_type']

FEATURES_TO_DUMMIES = ['linqmap_subtype', 'linqmap_city', 'linqmap_street', 'linqmap_roadType',
                        'linqmap_roadType', 'linqmap_roadType', 'linqmap_roadType', 'day_of_week', 'hour_in_day']

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


def convert_dates(df: pd.DataFrame) -> None:
    dts = pd.to_datetime(df['update_date'], unit='ms')
    df['update_date_new'] = [dt.date() for dt in dts]
    df['update_time'] = [dt.time() for dt in dts]
    df['hour_in_day'] = dts.dt.hour  # hour as int from 0 to 24
    df['day_of_week'] = dts.dt.dayofweek  # hour as int from 0 to 24


def convert_coordinates(data) -> None:
    X = pd.Series.to_numpy(data['x'])
    Y = pd.Series.to_numpy(data['y'])
    transformer = Transformer.from_crs("EPSG:6991", "EPSG:4326")
    wgs84_coords = [transformer.transform(X[i], Y[i]) for i in range(len(X))]
    data['y'] = [tup[0] for tup in wgs84_coords]
    data['x'] = [tup[1] for tup in wgs84_coords]


def remove_diluted_features(df: pd.DataFrame, diluted_proportion: float = .9) -> list:
    df.drop_duplicates(subset=['OBJECTID'], inplace=True)
    features = []
    n_samples = df.shape[0]
    for feature in df:
        num_empty_cell = df[feature].isnull().sum()
        if num_empty_cell / n_samples >= diluted_proportion:
            features.append(feature)
    features += ['OBJECTID', 'nComments']
    df.drop(FEATURES_TO_DROP, axis=1, inplace=True)
    return features


def geolocator(coordinates: str) -> str:
    """Return location coordinates and accurate address of the specified location."""
    geolocator1 = Nominatim(user_agent="tutorial", timeout=LOCATION_TIMEOUT)
    try:
        location = geolocator1.reverse(coordinates)
        if location is not None:
            return location.raw
    except GeocoderTimedOut as e:
        print(str(e))
    return ""


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


def process_accident(df: pd.DataFrame):
    # most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside the city,
    # put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)

    df['linqmap_subtype'].fillna(accident_type['linqmap_subtype'], inplace=True)


def process_road_closed(df):
    road_closed_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    null_road_closed_type = road_closed_type[road_closed_type['linqmap_subtype'].isna()]
    for idx, row in null_road_closed_type.iterrows():
        same_date = road_closed_type[road_closed_type["update_date_new"] == row['update_date_new']]
        if (not(row['linqmap_street'] is None)) and (same_date['linqmap_street'].str.contains(row['linqmap_street']).any()):
            df.at[idx,'linqmap_subtype'] = 'ROAD_CLOSED_CONSTRUCTION'
            continue
        row['linqmap_subtype'] = 'ROAD_CLOSED_EVENT'


def process_jam(df):
    jam_type = df[df["linqmap_type"] == "JAM"]
    null_jam_type = jam_type[jam_type['linqmap_subtype'].isna()]
    for idx, row in null_jam_type.iterrows():
        same_date_and_place = jam_type[(jam_type["update_date_new"] == row['update_date_new']) &
                                       (jam_type["linqmap_street"] == row['linqmap_street']) &
                                       (~(jam_type["linqmap_subtype"].isna()))]
        if same_date_and_place.empty:
            df.at[idx, 'linqmap_subtype'] = 'JAM_HEAVY_TRAFFIC'
            continue
        delta_time = same_date_and_place['update_date'].apply(lambda x: np.abs((x - row['update_date'])))
        closest = same_date_and_place.loc[delta_time.idxmin()]
        df.at[idx, 'linqmap_subtype'] = closest['linqmap_subtype']


def process_weatherhazard(df):
    weatherhazard_type = df[df["linqmap_type"] == "WEATHERHAZARD"]
    dist = weatherhazard_type.linqmap_subtype.value_counts(normalize=True)
    missing = weatherhazard_type['linqmap_subtype'].isnull()
    weatherhazard_type.loc[missing, 'linqmap_subtype'] = np.random.choice(dist.index,
                                                 size=len(weatherhazard_type[missing]),
                                                 p=dist.values)
    df['linqmap_subtype'].fillna(weatherhazard_type['linqmap_subtype'], inplace=True)


def process_city_street(df: pd.DataFrame, geo: bool) -> None:
    df['linqmap_street'].fillna(0, inplace=True)
    n_samples = df.shape[0]
    printProgressBar(0, n_samples, prefix='Preprocessing:', suffix='Complete', length=50)
    for i, (index, sample) in enumerate(df.iterrows()):
        curr_city = sample['linqmap_city']
        curr_street = sample['linqmap_street']
        found_district = False
        # update city district
        for district, cities in DISTRICTS_OF_ISRAEL.items():
            if curr_city in cities:
                df['linqmap_city'][index] = district
                found_district = True
                break
        if not found_district:
                df['linqmap_city'][index] = 'Out of district'

        # update street
        if curr_street != 0:
            road_numbers = re.findall("[0-9]+", curr_street)
            if len(road_numbers) > 0:
                df['linqmap_street'][index] = int(road_numbers[0])
            # else:
            #     iter = re.finditer('ל-', curr_street)
            #     indices = [m.start(0) for m in iter]
            #     if len(indices) > 0:
            #         curr_street = curr_street[::-1]
            #         curr_street = curr_street[:len(curr_street) - (indices[0] + 2)].strip()[::-1]
            #         df['linqmap_street'][index] = curr_street
            df['linqmap_street'][index] = "street"
        # city and street is missing (we search for the nearset coordinates and fill the missing data)
        if geo:
            if df['linqmap_city'][index] == 'Out of district' and df['linqmap_street'][index] == 0:
                x_y = str(df['y'][index]) + ', ' + str(df['x'][index])
                raw_address = geolocator(x_y)
                try:
                    geo_city = raw_address["address"]['city']
                except KeyError:
                    df['linqmap_city'][index] = 'Out of district'
                else:
                    found_district = False
                    for district, cities in DISTRICTS_OF_ISRAEL.items():
                        if geo_city in cities:
                            df['linqmap_city'][index] = district
                            found_district = True
                            break
                    if not found_district:
                        df['linqmap_city'][index] = 'Out of district'
                try:
                    geo_road = raw_address["address"]['road']
                except KeyError:
                    df['linqmap_street'][index] = 0
                else:
                    df['linqmap_street'][index] = geo_road

        printProgressBar(i + 1, n_samples, prefix='Preprocessing:', suffix='Complete', length=50)



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def make_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data=df, columns=FEATURES_TO_DUMMIES)


def preprocess(df: pd.DataFrame, geo: bool) -> pd.DataFrame:
    convert_dates(df)
    process_accident(df)
    process_road_closed(df)
    process_jam(df)
    process_weatherhazard(df)
    process_city_street(df, geo)
    remove_diluted_features(df)
    data = make_dummies(df)
    return data

