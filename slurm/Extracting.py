from pydantic import BaseModel, PrivateAttr
from typing import List, Optional, Tuple, Dict, ClassVar
from NcBase import NcBase
import pandas as pd
import numpy as np
import math
import logging
import gc
from os import path

class Extracting(BaseModel):
    region:Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    date_period:Tuple[str, str] = ('2020-01-01', '2020-12-31')
    spatial_resolution: float = 0.015625
    temporal_resolution: str = "1D"
    nc_morphology_dir: str =''
    nc_meteo_dir: str = ''
    nc_hydro_dir: str = ''
    latitude_name: str = 'lat'
    longitude_name: str = 'lon'
    date_name: str = 'date'
    _features: List[Dict] = PrivateAttr()
    _batch_size: int = PrivateAttr(default=500_000)
    _logger: logging.Logger = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        self._features = [
            {"alias": "clay",      "variable": "clay",  "lags": None,            "path": path.join(self.nc_morphology_dir, "clay.nc")},
            {"alias": "elevation", "variable": "elevation", "lags": None,        "path": path.join(self.nc_morphology_dir, "elevation.nc")},
            {"alias": "aspect",    "variable": "aspect", "lags": None,           "path": path.join(self.nc_morphology_dir, "aspect.nc")},
            {"alias": "slope",     "variable": "slope", "lags": None,            "path": path.join(self.nc_morphology_dir, "slope.nc")},
            {"alias": "sand",      "variable": "sand",  "lags": None,            "path": path.join(self.nc_morphology_dir, "sand.nc")},
            {"alias": "lc",        "variable": "lc",    "lags": None,            "path": path.join(self.nc_morphology_dir, "lc.nc")},
            {"alias": "pet",  "variable": "pet",  "lags": [1, 330, 365],         "path": path.join(self.nc_meteo_dir, "pet.nc")},
            {"alias": "pre",  "variable": "pre",  "lags": [90],                  "path": path.join(self.nc_meteo_dir, "pre.nc")},
            {"alias": "tavg", "variable": "tavg", "lags": [365],                 "path": path.join(self.nc_meteo_dir, "tavg.nc")},
            {"alias": "tmax", "variable": "tmax", "lags": [1, 14, 35, 42, 49, 56, 365], "path": path.join(self.nc_meteo_dir, "tmax.nc")},
            {"alias": "tmin", "variable": "tmin", "lags": [330, 365],             "path": path.join(self.nc_meteo_dir, "tmin.nc")},
            {"alias": "swc1", "variable": "SWC_L01", "lags": [330, 365],             "path": path.join(self.nc_hydro_dir, "SWC_L01.nc")},
            {"alias": "swc2", "variable": "SWC_L02", "lags": [365],                  "path": path.join(self.nc_hydro_dir, "SWC_L02.nc")},
            {"alias": "swc3", "variable": "SWC_L03", "lags": [330, 365],             "path": path.join(self.nc_hydro_dir, "SWC_L03.nc")},
            {"alias": "swc4", "variable": "SWC_L04", "lags": [365],                  "path": path.join(self.nc_hydro_dir, "SWC_L04.nc")},
            {"alias": "swc6", "variable": "SWC_L06", "lags": [180, 300, 365],             "path": path.join(self.nc_hydro_dir, "SWC_L06.nc")},
            {"alias": "LM_swc2", "variable": "SWC_L02", "lags": None,                "path": path.join(self.nc_hydro_dir, "LM_SWC_L02.nc")},
        ]

        self._set_logger()
        self.check_feature_files()

    def _set_logger(self):
        self._logger = logging.getLogger("Feature Extractor")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

    def get_coordinates(self, job_id:int, n:int) -> pd.DataFrame:
        min_lat = min(self.region[0], self.region[2])
        max_lat = max(self.region[0], self.region[2])
        min_lon = min(self.region[1], self.region[3])
        max_lon = max(self.region[1], self.region[3])
        min_date = min(self.date_period[0], self.date_period[1])
        max_date = max(self.date_period[0], self.date_period[1])

        latitudes = np.arange(min_lat, max_lat + self.spatial_resolution / 2, self.spatial_resolution)
        longitudes = np.arange(min_lon, max_lon + self.spatial_resolution / 2, self.spatial_resolution)
        dates = pd.date_range(start=min_date , end=max_date, freq=self.temporal_resolution, inclusive="both")

        total = len(latitudes) * len(longitudes) * len(dates)
        per_job = math.ceil(total / n)
        s_idx = job_id * per_job
        e_idx = min(s_idx + per_job, total)

        coords = []
        for i in range(s_idx, e_idx):
            date = dates[i // (len(latitudes) * len(longitudes))]
            lat = latitudes[(i // len(longitudes)) % len(latitudes)]
            lon = longitudes[i % len(longitudes)]
            coords.append([date, lat, lon])
        df = pd.DataFrame(coords, columns=[self.date_name, self.latitude_name, self.longitude_name])
        return df

    def get_features_data(self, job_id:int, n:int) -> pd.DataFrame:

        keys = [self.latitude_name, self.longitude_name, self.date_name]
        coordinates = self.get_coordinates(job_id, n)

        dfs = []
        for feature in self._features:
            nc = NcBase.model_validate(feature).set_batch_size(self._batch_size)
            df = nc.get_dataframe(coordinates)
            df = df.set_index(keys)
            dfs.append(df)
            del df, nc
            gc.collect()

        feature_data = pd.concat(dfs, axis=1).reset_index()
        for c in feature_data.columns:
            if feature_data[c].dtype == "float64":
                feature_data[c] = feature_data[c].astype("float32")

        self._logger.info(f"Feature data extraction for job ID {job_id} completed.")
        self._logger.info(f"Extracted feature data shape: {feature_data.shape}")
        return feature_data

    def check_feature_files(self):

        for feature in self._features:
            if not path.isfile(feature['path']):
                raise FileNotFoundError(f"Feature NetCDF file not found: {feature['path']}")

    def set_date_name(self, name: str):
        self.date_name = name

    def set_latitude_name(self, name: str):
        self.latitude_name = name

    def set_longitude_name(self, name: str):
        self.longitude_name = name

    def set_batch_size(self,  size: int):
        self._batch_size = size

