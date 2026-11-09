import xarray as xr
import numpy as np
import pandas as pd
from slurm.NcBase import NcBase

class TestNcBase:

    def test_aspect_netcdf(self, aspect_netcdf_file):
        ds = xr.open_dataset(aspect_netcdf_file)
        assert "aspect" in ds
        assert ds["aspect"].dims == ("lat", "lon")
        assert ds["aspect"].dtype == np.float32
        ds.close()

    def test_get_values_aspect(self,  aspect_netcdf_file, coordinates):
        feature = {"alias": "aspect", "variable": "aspect", "lags": None, "path": aspect_netcdf_file}
        lats = coordinates["lat"].to_numpy()
        lons = coordinates["lon"].to_numpy()
        nc = NcBase.model_validate(feature)
        results = nc.get_values(lats, lons, None)

        assert isinstance(results, np.ndarray)
        assert results.shape == (len(lats), 1)
        assert np.isfinite(results).all()

    def test_get_values_lm_tmin(self, lm_tmin_netcdf_file, coordinates):
        feature = {"alias": "lm_tmin", "variable": "tmin", "lags": None, "path": lm_tmin_netcdf_file}
        lats = coordinates["lat"].to_numpy()
        lons = coordinates["lon"].to_numpy()
        dates = coordinates["date"].to_numpy()
        nc = NcBase.model_validate(feature)
        results = nc.get_values(lats, lons, dates)

        assert isinstance(results, np.ndarray)
        assert results.shape == (len(lats), 1)
        assert np.isfinite(results).all()

    def test_get_values_tmin(self, tmin_netcdf_file, coordinates):
        lags = [1, 2, 3]
        feature = {"alias": "tmin", "variable": "tmin", "lags": lags, "path": tmin_netcdf_file}
        lats = coordinates["lat"].to_numpy()
        lons = coordinates["lon"].to_numpy()
        dates = coordinates["date"].to_numpy()
        nc = NcBase.model_validate(feature)
        results = nc.get_values(lats, lons, dates)
        spatial_size = len(np.unique(lats)) *  len(np.unique(lons))

        assert isinstance(results, np.ndarray)
        assert results.shape == (len(lats), 3)
        assert np.isfinite(results[:, 0]).sum() == len(lats) - (spatial_size * (lags[0] -1))
        assert np.isfinite(results[:, 1]).sum() == len(lats) - (spatial_size * (lags[1] -1))
        assert np.isfinite(results[:, 2]).sum() == len(lats) - (spatial_size * (lags[2] -1))

    def test_get_dataframe_aspect(self, aspect_netcdf_file, coordinates):
        feature = {"alias": "aspect", "variable": "aspect", "lags": None, "path": aspect_netcdf_file}
        nc = NcBase.model_validate(feature)
        nc.set_batch_size(10_000)
        df = nc.get_dataframe(coordinates)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == coordinates.shape[0]
        assert "aspect" in df.columns
        assert np.isfinite(df["aspect"]).all()

    def test_get_dataframe_lm_tmin(self,  lm_tmin_netcdf_file, coordinates):
        feature = {"alias": "lm_tmin", "variable": "tmin", "lags": None, "path": lm_tmin_netcdf_file}
        nc = NcBase.model_validate(feature)
        nc.set_batch_size(10_000)
        df = nc.get_dataframe(coordinates)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == coordinates.shape[0]
        assert "lm_tmin" in df.columns
        assert np.isfinite(df["lm_tmin"]).all()

    def test_get_dataframe_tmin(self, tmin_netcdf_file, coordinates):
        lags = [1, 2, 3]
        feature = {"alias": "tmin", "variable": "tmin", "lags": lags, "path": tmin_netcdf_file}
        nc = NcBase.model_validate(feature)
        nc.set_batch_size(10_000)
        df = nc.get_dataframe(coordinates)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == coordinates.shape[0]
        for lag in lags:
            col_name = f"tmin_{lag}"
            assert col_name in df.columns
            spatial_size = len(np.unique(coordinates["lat"])) *  len(np.unique(coordinates["lon"]))
            assert np.isfinite(df[col_name]).sum() == len(coordinates) - (spatial_size * (lag -1))

    def test_get_dataframe_tmin_no_batch(self, tmin_netcdf_file, coordinates):
            lags = [1, 2, 3]
            feature = {"alias": "tmin", "variable": "tmin", "lags": lags, "path": tmin_netcdf_file}
            nc = NcBase.model_validate(feature)
            nc.set_batch_size(1)
            coordinates = coordinates.head(50)
            df = nc.get_dataframe(coordinates)

            assert isinstance(df, pd.DataFrame)
            assert df.shape[0] == coordinates.shape[0]
            for lag in lags:
                col_name = f"tmin_{lag}"
                assert col_name in df.columns
