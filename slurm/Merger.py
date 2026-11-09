import logging
import pandas as pd
import numpy as np
import xarray as xr
import re
import dask.array as da
import dotenv
from netCDF4 import Dataset, date2num
import os
from pydantic import BaseModel, PrivateAttr
from typing import List

class Merger(BaseModel):

    _input_files:List[str] = PrivateAttr()
    _initialized: bool = PrivateAttr(default=False)
    _output_dir: str = PrivateAttr()
    _output_file_name: str = PrivateAttr(default='FEPO.nc')
    _latitude_name: str = PrivateAttr(default='lat')
    _latitude_alias: str = PrivateAttr(default='lat')
    _longitude_name: str = PrivateAttr(default='lon')
    _longitude_alias: str = PrivateAttr(default='lon')
    _date_name: str = PrivateAttr(default='sdate')
    _date_alias: str = PrivateAttr(default='date')
    _spec_name: str = PrivateAttr(default='spec_code')
    _spec_values: List[int] = PrivateAttr(default=[0,1,2,3])
    _species_map: dict = PrivateAttr(default={0: 'Phlebotomus papatasi', 1: 'ٖPhlebotomus perniciosus', 2: 'Phlebotomus sergenti', 3: 'Phlebotomus tobbi'})
    _value_name: str = PrivateAttr(default='pred')
    _time_units: str = PrivateAttr(default="seconds since 1970-01-01 00:00:00")
    _calendar: str = PrivateAttr(default="standard")
    _dtype: str = "float32"

    _logger: logging.Logger = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, input_dir, output_dir):
        super().__init__()
        self._output_dir = output_dir
        self._input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
        self._input_files = sorted(self._input_files)
        self._set_logger()
        self._validate_input_files()

    # Private Methods =================================================

    def _set_logger(self):
        self._logger = logging.getLogger("Merger")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

    def _validate_input_files(self):
        self._logger.info("Validating input files")

        nums = []
        pat = re.compile(r"^predictions_(\d+)\.csv$")
        for file in self._input_files:
            base = os.path.basename(file)
            m = pat.match(base)
            if not m:
                raise ValueError(f"Invalid file name format: {base}. Expected 'predictions_<number>.csv'.")

            nums.append(int(m.group(1)))

        expected = set(range(len(self._input_files)))
        seen = set(nums)

        missing = sorted(expected - seen)
        extra   = sorted(seen - expected)

        if missing:
            raise ValueError(f"Apparently there is a missing file: {missing[0]:05d}.csv (or predictions_{missing[0]:05d}.csv)")

        if len(nums) != len(seen):
            raise ValueError("Duplicate index in file names detected.")
        if extra:
            raise ValueError(f"File index out of expected range: {extra}")

        self._logger.info("Input files validated")

    def _write_to_netcdf_file(self, buffer: pd.DataFrame):

        if not self._initialized:
            self._initialized = True
            n_points = buffer[[self._latitude_name, self._longitude_name]].drop_duplicates().shape[0]
            self._init_netcdf_file(n_points)
        self._append_to_netcdf(buffer)

    def _init_netcdf_file(self, n_points: int):
        nc = Dataset(self.get_temp_file_name(), "w", format="NETCDF4")
        nc.createDimension("date", None)         # unlimited
        nc.createDimension("point", n_points)

        nc.createVariable(self._date_alias, "f8", ("date",))
        nc.createVariable(self._latitude_alias, "f4", ("point",))
        nc.createVariable(self._longitude_alias, "f4", ("point",))

        for code in self._spec_values:
            var = f"sp_{code:02d}"
            nc.createVariable(
                var, "f4", ("date","point"),
                zlib=True, complevel=1, fill_value=np.float32(np.nan)
            )

        nc.close()

    def _append_to_netcdf(self,  buffer: pd.DataFrame):

        days = buffer[self._date_name].dt.floor("D").unique()
        with Dataset(self.get_temp_file_name(), "a") as nc:

            for day in days:
                self._logger.info(f"Writing {day} to netcdf")
                day_num = date2num(pd.Timestamp(day).to_pydatetime(), units=self._time_units, calendar=self._calendar)
                buffer_day = buffer[buffer[self._date_name].dt.floor("D") == day]
                i = nc.dimensions[self._date_alias].size
                if i == 0:
                    coords = buffer_day[[self._latitude_name, self._longitude_name]].drop_duplicates().reset_index(drop=True)
                    nc.variables[self._latitude_alias][:] = coords[self._latitude_name].values.astype(self._dtype, copy=False)
                    nc.variables[self._longitude_alias][:] = coords[self._longitude_name].values.astype(self._dtype, copy=False)

                nc.variables[self._date_alias][i] = day_num
                n_points = nc.dimensions["point"].size
                for code in self._spec_values:
                    var_name = f"sp_{code:02d}"
                    mask = buffer_day[self._spec_name] == code
                    values = buffer_day.loc[mask, self._value_name].values
                    if len(values) != n_points:
                        raise ValueError(f"Data points for spec_code {code} on {day} do not match the number of grid points ({len(values)} vs {n_points}).")
                    nc.variables[var_name][i, :] = values.astype(self._dtype, copy=False)

    def _finalized_output(self):

        self._logger.info(f"Finalizing output to {self.get_output_file_path()}")
        self._logger.info("It may take a while...")

        ds = xr.open_dataset(self.get_temp_file_name(), chunks={self._date_alias: 1}, decode_times=False)
        lat_p = ds[self._latitude_alias].values
        lon_p = ds[self._longitude_alias].values
        lats = np.unique(lat_p)
        lons = np.unique(lon_p)

        nlat, nlon = len(lats), len(lons)

        if nlat * nlon != ds.sizes["point"]:
            raise ValueError(f"Not full grid: {nlat}*{nlon} != {ds.sizes['point']}")
        order = np.lexsort((lon_p, lat_p)).astype(np.int64)


        def to_lonlat(darr_2d):
            tmp = da.take(darr_2d, order, axis=1)
            return tmp.reshape((tmp.shape[0], nlat, nlon))

        out = xr.Dataset(coords={self._date_alias: ds[self._date_alias], self._latitude_alias: lats, self._longitude_alias: lons})
        out[self._latitude_alias].attrs.update({"standard_name":"latitude","units":"degrees_north", "axis": "Y"})
        out[self._longitude_alias].attrs.update({"standard_name":"longitude","units":"degrees_east", "axis": "X"})
        out[self._date_alias].attrs.update({
            "units": self._time_units,
            "calendar": self._calendar,
            "standard_name": "time",
            "long_name": "time",
            "axis": "T",
        })

        for v in [k for k in ds.data_vars if k.startswith("sp_")]:
            out[v] = ((self._date_alias,self._latitude_alias,self._longitude_alias), to_lonlat(ds[v].data))
            out[v].attrs.update(ds[v].attrs)
            code = int(v.split("_")[1])
            if code in self._species_map:
                out[v].attrs["species"] = self._species_map[code]
                out[v].attrs["long_name"] = f"Daily density for {self._species_map[code]}"
                out[v].attrs["description"] = f"Estimated density of {self._species_map[code]}"


        desired = {self._date_alias: 1, self._latitude_alias: 100, self._longitude_alias: 100}
        encoding = {}
        for v in out.data_vars:
            dims = out[v].dims
            chunks = tuple(min(desired[d], out.sizes[d]) for d in dims)
            encoding[v] = {"zlib": True, "complevel": 1, "dtype": self._dtype, "chunksizes": chunks}


        # global attrs
        out.attrs["Conventions"] = "CF-1.8"
        out.attrs["title"] = "FEPO daily predictions"
        out.attrs["institution"] = "Helmholtz Centre for Environmental Research (UFZ)"
        out.attrs["creator_name"] = "Dr. Majid Soheili, Dr. Ehsan Modiri, Dr. Oldrich Rakovec, Prof. Dr. rer. nat. habil. Luis E. Samaniego Eguiguren"

        task = out.to_netcdf(self.get_output_file_path(), engine="netcdf4", format="NETCDF4", encoding=encoding, compute=False)
        task.compute()

        self._logger.info("Removing temporary file")
        os.remove(self.get_temp_file_name())

    def run(self):
        buffer = pd.DataFrame()
        current_date = None
        for file in self._input_files:
            self._logger.info(f"Reading {file}")
            df = pd.read_csv(file, parse_dates=[self._date_name])
            buffer = pd.concat([buffer, df], ignore_index=True)
            max_date_in_file = buffer[self._date_name].max()

            if current_date is None:
                current_date = buffer[self._date_name].min()

            if max_date_in_file > current_date:
                to_process = buffer[buffer[self._date_name] < max_date_in_file]
                self._write_to_netcdf_file(to_process)
                buffer = buffer[buffer[self._date_name] >= max_date_in_file]
                current_date = max_date_in_file

        if not buffer.empty:
            self._logger.info("Writing remaining (last day) data to netcdf")
            self._write_to_netcdf_file(buffer)

        self._finalized_output()
        self._logger.info("Finished")

   # Public Properties =========================================
    def get_output_file_path(self)  -> str:
        return os.path.join(self._output_dir, self._output_file_name)
    def get_temp_file_name(self) -> str:
        return os.path.join(self._output_dir, "temp_FEPO.nc")


if __name__ == "__main__":

    dotenv.load_dotenv()
    artifact_dir= os.getenv("ARTIFACT_DIR")

    input_dir = artifact_dir
    output_dir = artifact_dir

    merger = Merger(input_dir=input_dir, output_dir=output_dir)
    merger.run()