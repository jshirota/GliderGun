import re
from pathlib import Path
from typing import Union

import numpy as np
from numpy import ndarray
from rasterio.crs import CRS


def create_directory(file_path: str):
    directory = "/".join(re.split(r"/|\\", file_path)[0:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_crs(crs: Union[int, CRS]):
    return CRS.from_epsg(crs) if isinstance(crs, int) else crs


def format_type(data: ndarray):
    if data.dtype == "float64":
        return np.asanyarray(data, dtype="float32")
    if data.dtype == "int64":
        return np.asanyarray(data, dtype="int32")
    if data.dtype == "uint64":
        return np.asanyarray(data, dtype="uint32")
    return data


def get_nodata_value(dtype: str) -> Union[float, int, None]:
    if dtype == "bool":
        return None
    if dtype.startswith("float"):
        return float(np.finfo(dtype).min)
    if dtype.startswith("uint"):
        return np.iinfo(dtype).max
    return np.iinfo(dtype).min
