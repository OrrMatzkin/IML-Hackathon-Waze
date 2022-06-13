from geopy import Nominatim
from geopy.distance import distance
from geopy.exc import GeocoderTimedOut
from pyproj import Transformer
import typing
import copy
import pandas as pd
import numpy as np
import re

FEATURES_TO_DROP_T1 = ['linqmap_reportDescription', 'linqmap_nearby', 'update_time', 'linqmap_street',
                    'linqmap_expectedBeginDate', 'linqmap_expectedEndDate', 'OBJECTID', 'nComments',
                    'linqmap_reportMood', 'linqmap_magvar', 'update_date_new',
                    'pubDate']


FEATURES_TO_DROP_T2 = ['linqmap_reportDescription', 'linqmap_nearby',
                    'update_date', 'linqmap_street',
                    'linqmap_expectedBeginDate', 'linqmap_expectedEndDate',
                    'OBJECTID', 'nComments',
                    'linqmap_reportMood', 'linqmap_magvar', 'pub_date',
                    'pub_time']

FEATURES_TO_DUMMIES_T1 = ['linqmap_subtype', 'linqmap_type', 'linqmap_city',
                       'linqmap_roadType', 'day_of_week', 'hour_in_day']

FEATURES_TO_DUMMIES_T2 = ['linqmap_city', 'linqmap_roadType']

FEATURES_TO_CATEGORIZE = ['linqmap_subtype', 'linqmap_type', 'linqmap_street', 'linqmap_city']

DISTRICTS_OF_ISRAEL = {"North District": ['בית שאן', 'טבריה', 'טמרה', 'יקנעם עלית', 'כרמיאל', 'מגדל העמק',
                                          "מע'אר", 'מעלות תרשיחא', 'נהריה', 'נוף הגליל', 'נצרת', "סחנין",
                                          'עראבה', 'עכו',
                                          'עפולה', 'צפת', 'קריית שמונה', 'שפרעם'],
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

EVENT_TYPES = {'ACCIDENT': 0, 'JAM': 1, 'ROAD_CLOSED': 2, "WEATHERHAZARD": 3}

EVENT_SUBTYPES = {'ACCIDENT_MAJOR': 0,
                  'ACCIDENT_MINOR': 1,
                  'ROAD_CLOSED_CONSTRUCTION': 2,
                  "ROAD_CLOSED_EVENT": 3,
                  'JAM_HEAVY_TRAFFIC': 4,
                  'JAM_MODERATE_TRAFFIC': 5,
                  'JAM_STAND_STILL_TRAFFIC': 6,
                  'HAZARD_ON_ROAD': 7,
                  'HAZARD_ON_ROAD_CAR_STOPPED': 8,
                  'HAZARD_ON_ROAD_CONSTRUCTION': 9,
                  'HAZARD_ON_ROAD_ICE': 10,
                  'HAZARD_ON_ROAD_OBJECT': 11,
                  'HAZARD_ON_ROAD_POT_HOLE': 12,
                  'HAZARD_ON_ROAD_ROAD_KILL': 13,
                  'HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT': 14,
                  'HAZARD_ON_SHOULDER': 15,
                  'HAZARD_ON_SHOULDER_ANIMALS': 16,
                  'HAZARD_ON_SHOULDER_CAR_STOPPED': 17,
                  'HAZARD_ON_SHOULDER_MISSING_SIGN': 18,
                  'HAZARD_WEATHER': 19,
                  'HAZARD_WEATHER_FLOOD': 20,
                  'HAZARD_WEATHER_FOG': 21,
                  'HAZARD_WEATHER_HAIL': 22,
                  'HAZARD_WEATHER_HEAVY_SNOW': 23
                  }


def process_update_date_T1(df: pd.DataFrame) -> None:
    """
    Converts for task 1 the update_date feature to dt.data and creates 3 more features:
    update_time, hour_in_day (0-23), day_of_week (0-6).
    :param df: data frame.
    :return: None.
    """
    dts = pd.to_datetime(df['update_date'], unit='ms')
    df['update_date_new'] = [dt.date() for dt in dts]
    df['update_time'] = [dt.time() for dt in dts]
    df['hour_in_day'] = dts.dt.hour  # hour as int from 0 to 23
    df['day_of_week'] = dts.dt.dayofweek  # day of week as int from 0 to 6


def process_update_date_T2(df: pd.DataFrame) -> None:
    """
    Converts for task 2 the update_date feature to dt.data and creates 3 more features:
    update_time, hour_in_day (0-23), day_of_week (0-6).
    :param df: data frame.
    :return: None.
    """
    dts = pd.to_datetime(df['pubDate']).dt.tz_localize('UTC').dt.tz_convert('Israel')
    df['pub_date'] = [dt.date() for dt in dts]
    df['pub_time'] = [dt.time() for dt in dts]
    df['hour_in_day'] = dts.dt.hour  # hour as int from 0 to 23
    df['day_of_week'] = dts.dt.dayofweek  # day of week as int from 0 to 6

    dts = pd.to_datetime(df['update_date']).dt.tz_localize('UTC').dt.tz_convert('Israel')
    df['update_date_new'] = [dt.date() for dt in dts]


def convert_coordinates(df: pd.DataFrame) -> None:
    """
    Converts coordinates to WGS84.
    :param df: data frame.
    :return: None.
    """
    X = pd.Series.to_numpy(df['x'])
    Y = pd.Series.to_numpy(df['y'])
    transformer = Transformer.from_crs("EPSG:6991", "EPSG:4326")
    wgs84_coords = [transformer.transform(X[i], Y[i]) for i in range(len(X))]
    df['y'] = [tup[0] for tup in wgs84_coords]
    df['x'] = [tup[1] for tup in wgs84_coords]


def compress_4_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compress every 4 samples (rows) in the data frame into a single sample.
    :param df: data frame.
    :return: compressed data frame.
    """
    # make list of sub lists with 4 samples
    list_of_4_rows = [[df.iloc[i + j, :] for j in range(4)] for i in range(df.shape[0] - 4)]
    # concat each sub list to become one sample
    train_X = pd.DataFrame([pd.concat(four_rows, axis=0, ignore_index=True) for four_rows in list_of_4_rows])
    return train_X


def categorize_features(df: pd.DataFrame) -> None:
    """
    Obtaining a numeric representation for the FEATURES_TO_CATEGORIZE.
    :param df: data frame
    :return: None
    """
    for feature in FEATURES_TO_CATEGORIZE:
        df[feature] = pd.factorize(df[feature])[0] + 1


def remove_diluted_features(df: pd.DataFrame, task: int, diluted_proportion: float = .9) -> list:
    """
    Removes features that do not have enough data, that there diluted is more the diluted proportion
    and duplicated samples (by id).
    :param df: data frame.
    :param diluted_proportion: diluted proportion (default value is 0.9).
    :return: the names of the removes features.
    """
    df.drop_duplicates(subset=['OBJECTID'], inplace=True)
    features = []
    n_samples = df.shape[0]
    for feature in df:
        num_empty_cell = df[feature].isnull().sum()
        if num_empty_cell / n_samples >= diluted_proportion:
            features.append(feature)
    features += ['OBJECTID', 'nComments']
    if task == 1:
        df.drop(FEATURES_TO_DROP_T1, axis=1, inplace=True)
    elif task == 2:
        df.drop(FEATURES_TO_DROP_T2, axis=1, inplace=True)
    return features


def geo_locator(coordinates: str) -> str:
    """
    Return location coordinates and accurate address of the specified location
    :param coordinates: coordinates of thje location
    :return: if founded the info about the location (city, street, road name, etc...)
    """
    geolocator1 = Nominatim(user_agent="tutorial", timeout=LOCATION_TIMEOUT)
    try:
        location = geolocator1.reverse(coordinates)
        if location is not None:
            return location.raw
    except GeocoderTimedOut as e:
        print(str(e))
    return ""


def get_nearest_location(x: float, y: float, df: pd.DataFrame) -> typing.Tuple[str, str]:
    """
    Finds the nearset location (city and street) of the given (x,y) coordinates inside the data.
    :param x: x coordinate.
    :param y: y coordinate.
    :param df: data frame.
    :return: the nearset city and street from the data
    """
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


def process_accident(df: pd.DataFrame) -> None:
    """
     Most major accidents happened outside of city, so if 'linqmap_subtype' is null and is outside the city,
     put 'ACCIDENT_MAJOR', and 'ACCIDENT_MINOR' otherwise.
    :param df: data frame.
    :return: None.
    """
    accident_type = df[df["linqmap_type"] == "ACCIDENT"]
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & accident_type["linqmap_city"].isna(), 'ACCIDENT_MAJOR', inplace=True)
    accident_type['linqmap_subtype'].mask(
        accident_type['linqmap_subtype'].isna() & ~accident_type["linqmap_city"].isna(), 'ACCIDENT_MINOR', inplace=True)
    df['linqmap_subtype'].fillna(accident_type['linqmap_subtype'], inplace=True)


def process_road_closed(df: pd.DataFrame) -> None:
    """
    Fill missing data in the subtype feature in the case the type is road closure.
    If their another record in the same day and on the same road of subtype ROAD_CLOSED_CONSTRUCTION it fills it as
    ROAD_CLOSED_CONSTRUCTION, else it fills it as ROAD_CLOSED_EVENT.
    :param df: data frame.
    :return: None.
    """
    road_closed_type = df[df["linqmap_type"] == "ROAD_CLOSED"]
    null_road_closed_type = road_closed_type[road_closed_type['linqmap_subtype'].isna()]
    for idx, row in null_road_closed_type.iterrows():
        same_date = road_closed_type[road_closed_type["update_date_new"] == row['update_date_new']]
        if (not (row['linqmap_street'] is None)) and (
                same_date['linqmap_street'].str.contains(row['linqmap_street']).any()):
            df.at[idx, 'linqmap_subtype'] = 'ROAD_CLOSED_CONSTRUCTION'
            continue
        row['linqmap_subtype'] = 'ROAD_CLOSED_EVENT'


def process_jam(df: pd.DataFrame) -> None:
    """
    Fills missing data in the subtype feature if the event type is JAM.
    We look for another event JAM record in the data in the same day and road, if we found we fill the missing
    data just like the found record, else we fill JAM_HEAVY_TRAFFIC (the middle option for a JAM).
    :param df: data frame.
    :return: None.
    """
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


def process_weatherhazard(df: pd.DataFrame) -> None:
    """
    We calculted the distribution of all WEATHERHAZARD type event, and filled the subtype with random subtypes
    while maintaining the same distribution.
    :param df: data frame.
    :return: None.
    """
    weatherhazard_type = df[df["linqmap_type"] == "WEATHERHAZARD"]
    dist = weatherhazard_type.linqmap_subtype.value_counts(normalize=True)
    missing = weatherhazard_type['linqmap_subtype'].isnull()
    if missing.empty:
        return
    weatherhazard_type.loc[missing, 'linqmap_subtype'] = np.random.choice(dist.index,
                                                                          size=len(weatherhazard_type[missing]),
                                                                          p=dist.values)
    df['linqmap_subtype'].fillna(weatherhazard_type['linqmap_subtype'], inplace=True)


def process_city_street(df: pd.DataFrame, name: str, bar: bool = True) -> None:
    """
    Categorizes all cities into different districts and if the street name is actuality a road number extracts the
    number and converts it to an integer.
    :param df: data frame.
    :param name: text to print in the progress bar.
    :param bar: flag to print progress bar.
    :return: None.
    """
    df['linqmap_street'].fillna(0, inplace=True)
    n_samples = df.shape[0]
    if bar:
        print_progress_bar(0, n_samples, prefix=f'Preprocessing {name} data:', suffix='Complete', length=50)
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
            df['linqmap_street'][index] = "street"
        if bar:
            print_progress_bar(i + 1, n_samples, prefix=f'Preprocessing {name} data:', suffix='Complete', length=50)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
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
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()

def make_dummies_T1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the need dummies features.
    If needed adds a zero feature for some dummies.
    :param df: data frame.
    :return: data frame with dummies (instead of regular features).
    """
    data = pd.get_dummies(data=df, columns=FEATURES_TO_DUMMIES_T1)
    start_str = "day_of_week_"
    for i in range(7):
        title = start_str + str(i)
        if title not in df.columns:
            data[title] = np.zeros(df.shape[0]).astype(int)
    start_str = "hour_in_day_"
    for i in range(24):
        title = start_str + str(i)
        if title not in data.columns:
            data[title] = np.zeros(df.shape[0]).astype(int)
    start_str = "linqmap_roadType_"
    for i in range(23):
        title = start_str + str(i)
        if title not in data.columns:
            data[title] = np.zeros(df.shape[0]).astype(int)
    start_str = "linqmap_type_"
    for i in EVENT_TYPES.keys():
        title = start_str + str(i)
        if title not in data.columns:
            data[title] = np.zeros(df.shape[0]).astype(int)
    start_str = "linqmap_subtype_"
    for i in EVENT_SUBTYPES.keys():
        title = start_str + str(i)
        if title not in data.columns:
            data[title] = np.zeros(df.shape[0]).astype(int)
    start_str = "linqmap_city_"
    for i in DISTRICTS_OF_ISRAEL.keys():
        title = start_str + i
        if title not in data.columns:
            data[title] = np.zeros(df.shape[0]).astype(int)
    if 'linqmap_city_Out of district' not in df.columns:
        data['linqmap_city_Out of district'] = np.zeros(df.shape[0]).astype(int)
    return data


def make_dummies_T2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the needed dummies features for task 2.
    :param df: data frame.
    :return: data frame with dummies (instead of regular features).
    """
    return pd.get_dummies(data=df, columns=FEATURES_TO_DUMMIES_T2)


def preprocess_task1(df: pd.DataFrame, bar_name: str) -> pd.DataFrame:
    """
    The entire Preprocess stage for task 1.
    :param df: raw data frame.
    :param bar_name: text to show in the progress bar.
    :return: data frame after preprocess.
    """
    process_update_date_T1(df)
    process_accident(df)
    process_road_closed(df)
    process_jam(df)
    process_weatherhazard(df)
    process_city_street(df, bar_name)
    remove_diluted_features(df, 1)
    types_col = df["linqmap_type"]
    subtypes_col = df["linqmap_subtype"]
    dummies_data = make_dummies_T1(df)
    dummies_data["linqmap_type"] = types_col
    dummies_data["linqmap_subtype"] = subtypes_col
    return dummies_data


def preprocess_task2(df: pd.DataFrame, geo: bool):
    """
    The entire Preprocess stage for task 2.
    :param df: raw data frame.
    :param geo: text to show in the progress bar.
    :return: data frame after preprocess.
    """
    data = copy.deepcopy(df)
    process_update_date_T2(data)
    process_accident(data)
    process_road_closed(data)
    process_jam(data)
    process_weatherhazard(data)
    process_city_street(data, geo)
    remove_diluted_features(data, 2)
    hours = data['hour_in_day']
    days = data['day_of_week']
    data = data.drop(['hour_in_day', 'day_of_week'], axis=1)
    data = make_dummies_T2(data)
    data['hour_in_day'], data['day_of_week'] = hours, days
    return data
