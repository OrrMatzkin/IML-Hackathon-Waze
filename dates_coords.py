from datetime import datetime
import pandas as pd
from pyproj import CRS
from pyproj import Transformer


def convert_dates(data: pd.DataFrame) -> None:
    dates = data['update_date']
    data['update_date'] = pd.Series(
        [datetime.fromtimestamp(time) for time in dates])


def convert_coordinates(data) -> None:
    X = data['x']
    Y = data['y']

    crs = CRS.from_epsg(6991)
    crs.to_epsg()
    crs = CRS.from_proj4(
        "+proj=tmerc +lat_0=31.7343936111111 +lon_0=35.2045169444445 "
        "+k=1.0000067 +x_0=219529.584 +y_0=626907.39 +ellps=GRS80 "
        "+towgs84=-24.002400,-17.103200,-17.844400,-0.33007,-1.852690,"
        "1.669690,5.424800 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:6991", "EPSG:4326")
    wgs84_coords = [transformer.transform(X[i], Y[i]) for i in range(len(X))]
    X, Y = wgs84_coords
