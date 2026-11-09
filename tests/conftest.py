import numpy as np
import xarray as xr
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def seed() -> int:
    return 42

@pytest.fixture(scope="session")
def spatial_grid() -> tuple[np.ndarray, np.ndarray]:
    """
    Shared spatial grid for morphology and spatial NetCDF tests.
    """
    lat = np.linspace(20.0, 21.0, 10)
    lon = np.linspace(-20.0, -19.0, 20)
    return lat, lon

@pytest.fixture(scope="session")
def temporal_grid() -> pd.DatetimeIndex:
    """
    Shared temporal grid for meteorological and hydrological NetCDF tests.
    """
    dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
    return dates

@pytest.fixture(scope="session")
def aspect_netcdf_file(tmp_path_factory, spatial_grid, seed) -> str:

    tmp_path = tmp_path_factory.mktemp("data")
    file_path = tmp_path / "morphology.nc"
    latitudes , longitudes = spatial_grid
    rng = np.random.default_rng(seed)
    data = rng.random((len(latitudes), len(longitudes))).astype("float32")
    ds = xr.Dataset({"aspect": (("lat", "lon"), data)}, coords={"lat": latitudes,"lon": longitudes})
    ds["aspect"].attrs.update({"long_name": "terrain aspect", "units": "1"})
    ds.attrs["Conventions"] = "CF-1.8"

    ds.to_netcdf(file_path)
    return str(file_path)

@pytest.fixture(scope="session")
def lm_tmin_netcdf_file(tmp_path_factory, spatial_grid, seed) -> str:

    tmp_path = tmp_path_factory.mktemp("data")
    file_path = tmp_path / "lm_tmin_netcdf_file.nc"
    latitudes , longitudes = spatial_grid
    month = np.arange(1, 13)
    rng = np.random.default_rng(seed)
    data = rng.random((len(month), len(latitudes), len(longitudes))).astype("float32")
    ds = xr.Dataset({"tmin": (("month", "lat", "lon"), data)},coords={"month": month,"lat": latitudes,"lon": longitudes})
    ds["tmin"].attrs.update({"long_name": "Long Monthly Minimum Temperature", "units": "1"})
    ds.attrs["Conventions"] = "CF-1.8"

    ds.to_netcdf(file_path)
    return str(file_path)

@pytest.fixture(scope="session")
def tmin_netcdf_file(tmp_path_factory, spatial_grid, temporal_grid, seed) -> str:

    tmp_path = tmp_path_factory.mktemp("data")
    file_path = tmp_path / "tmin_netcdf_file.nc"
    latitudes , longitudes = spatial_grid
    dates = temporal_grid
    rng = np.random.default_rng(seed)
    data = rng.random((len(dates), len(latitudes), len(longitudes))).astype("float32")
    ds = xr.Dataset({"tmin": (("date", "lat", "lon"), data)},coords={"date": dates,"lat": latitudes,"lon": longitudes})
    ds["tmin"].attrs.update({"long_name": "Daily Minimum Temperature", "units": "1"})
    ds.attrs["Conventions"] = "CF-1.8"

    ds.to_netcdf(file_path)
    return str(file_path)

@pytest.fixture(scope="session")
def coordinates(spatial_grid, temporal_grid) -> pd.DataFrame:
    latitudes , longitudes = spatial_grid
    dates = temporal_grid
    coords = []
    for date in dates:
        for lat in latitudes:
            for lon in longitudes:
                coords.append([date, lat, lon])
    df = pd.DataFrame(coords, columns=["date", "lat", "lon"])
    return df