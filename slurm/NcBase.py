import numpy as np
import xarray as xr
from netCDF4 import Dataset
import pandas as pd
import time
import logging
from typing import List, Optional, Tuple
from pydantic import BaseModel, PrivateAttr

class NcBase(BaseModel):
    path: str
    variable: str
    alias: Optional[str] = None
    lags: Optional[List[int]] = None
    latitude_name: Optional[str] = None
    longitude_name: Optional[str] = None
    date_name: Optional[str] = None
    chunk_size: Optional[Tuple[int, ...]] = None
    ds: Optional[xr.Dataset] = None
    _verbose: int = PrivateAttr(default=100_000)
    _batch_size: int = PrivateAttr(default=500_000)
    _logger: logging.Logger = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        self._set_logger()
        self._get_attributes()
        self._get_chunk_size()

    def _get_attributes(self):

        self.ds = xr.open_dataset(self.path, chunks="auto")
        context = {var: idx for idx, var in enumerate(self.ds.variables)}
        self.latitude_name = self._get_names('lat', context)
        self.longitude_name = self._get_names('lon', context)
        self.date_name = self._get_names('date', context)

        if self.is_date_required():
            if np.issubdtype(self.ds[self.date_name].dtype, np.datetime64):
                self._logger.info(f"Normalizing time coordinate '{self.date_name}' to daily resolution (dropping hours)." )
                self.ds = self.ds.assign_coords({self.date_name: self.ds[self.date_name].dt.floor("D")})

        self._logger.info(f"Detected NetCDF variables -> latitude: {self.latitude_name}, longitude: {self.longitude_name}, date: {self.date_name}")

    def _set_logger(self):
        self._logger = logging.getLogger("NetCDF Handler")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)
        self._logger.info("\nLogging configured in NcBase.")

    def _get_chunk_size(self):
        with Dataset(self.path, "r") as nc:
            var = nc.variables[self.variable]
            chunk_info = var.chunking()

            if chunk_info == 'contiguous':
                self.chunk_size=tuple(var.shape)
            else:
                self.chunk_size=tuple(chunk_info)

            self._logger.info(f"Chunk size: {self.chunk_size}")

    @staticmethod
    def _get_names(subject:str, context:dict):

        possibles=[]
        if subject == 'lat':
            possibles = ['lat', 'latitude', 'Latitude']
        elif subject == 'lon':
            possibles = ['lon', 'longitude', 'Longitude', 'LON']
        elif subject == 'date':
            possibles = ['time', 'date', 'DATE', 'Time', 'sdate', 'month']

        name = next((var for var in possibles if var in context), None)
        return name

    def get_values(self, latitudes:np.ndarray, longitudes:np.ndarray, dates:np.ndarray | None = None) -> np.ndarray:

        assert len(latitudes) == len(longitudes)
        if self.is_date_required():
            if dates is None: raise ValueError(f"Missing {self.date_name}.")
            assert len(dates) == len(latitudes)

        latitudes = xr.DataArray(np.asarray(latitudes, dtype=float).ravel(), dims=['points'])
        longitudes = xr.DataArray(np.asarray(longitudes, dtype=float).ravel(), dims=['points'])
        spatial_da = self.ds[self.variable].sel({self.latitude_name: latitudes, self.longitude_name: longitudes}, method='nearest')

        if self.is_date_required():
            if self.is_month_required():
                months = pd.to_datetime(dates).month
                months = xr.DataArray(np.asarray(months, dtype=int).ravel(), dims=['points'])
                values = spatial_da.sel({self.date_name: months}).compute().values
            else:
                max_lag = max(self.lags)
                dates = pd.to_datetime(dates)
                start_dates = dates.min() - pd.Timedelta(days=max_lag)
                end_dates = dates.max()
                dates = xr.DataArray(pd.to_datetime(dates).to_numpy(), dims=['points'])
                temporal_da = spatial_da.sel({self.date_name: slice(start_dates, end_dates)})
                per_lags = []
                for lag in self.lags:
                    rolling = temporal_da.rolling({self.date_name: lag}, min_periods=lag).mean()
                    vals = rolling.sel({self.date_name: dates})
                    per_lags.append(vals)
                values = xr.concat(per_lags, dim='lag').transpose('points', 'lag').compute().values
        else:
            values = spatial_da.compute().values

        if values.ndim == 1 :
            values = values.reshape(-1, 1)
        return values

    def get_dataframe(self, coordinates: pd.DataFrame) -> pd.DataFrame:

        self._logger.info(f"Starting extraction of values for variable {self.variable}")

        context = {col: idx for idx, col in enumerate(coordinates.columns)}
        lat_col = self._get_names('lat', context)
        lon_col = self._get_names('lon', context)
        date_col = self._get_names('date', context) or ''

        if lat_col is None:
            raise ValueError("Latitude column not found in the coordinates DataFrame.")
        if lon_col is None:
            raise ValueError("Longitude column not found in the coordinates DataFrame.")
        if self.is_date_required() and not date_col:
            raise ValueError("Date column not found in the coordinates DataFrame.")


        headers = self.get_headers()
        total_coords = len(coordinates)
        total_cols = len(headers)
        data = np.full((total_coords, total_cols), np.nan, dtype=np.float32)

        start = time.time()
        for s in range(0, total_coords, self._batch_size):
            start_time = time.time()
            e = min(s + self._batch_size, total_coords)
            chunk = coordinates.iloc[s:e]
            latitudes = chunk[lat_col].to_numpy()
            longitudes = chunk[lon_col].to_numpy()
            dates = None if not date_col else chunk[date_col].to_numpy()
            values = self.get_values(latitudes, longitudes, dates)
            data[s:e, :] = values.astype(np.float32, copy=False)
            self._logger.info(f"Processed records {s + 1} to {e} out of {total_coords} and took {time.time() - start_time:.2f} seconds.")


        df = pd.DataFrame(data, columns=headers)
        df.insert(0, lat_col, coordinates[lat_col].values)
        df.insert(1, lon_col, coordinates[lon_col].values)
        if date_col:
            df.insert(2, date_col, coordinates[date_col].values)

        self._logger.info(f"Extracting values of variable {self.variable} is finished. Total time: {time.time() - start:.2f} seconds / Total records: {total_coords}.")
        self.ds.close()

        return df

    def get_headers(self, prefix: str = None):
        prefix = prefix or ""
        variable_alias = self.alias or self.variable
        prefix += f"{variable_alias}"
        if self.lags is None:
            return [prefix]
        return [f"{prefix}_{lag}" for lag in self.lags]

    def is_date_required(self):
        return self.date_name is not None
    def is_month_required(self):
        return self.date_name == 'month'
    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size
        return self
    def set_verbose(self, verbose: int):
        self._verbose = verbose
        return self