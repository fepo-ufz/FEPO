import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import joblib
import os
import geopandas as gpd
from shapely.geometry import Point
# KDTree for longitude/latitude: https://stackoverflow.com/questions/10549402/kdtree-for-longitude-latitude

# We will have two approaches for pruning:
# 1: based on the training coordinates
# 2: based on the Eduardo maps

class FeSpecies:
    _model_dir = ""
    _raw_data_dir = ""
    _processed_data_dir = ""
    __name = "FeSpecies"

    def __init__(self, raw_data_dir: str = "", data_dir: str = ""):

        self.data: pd.DataFrame = pd.DataFrame()
        self.species = None
        self.species_count = None
        self.species_index = None
        self.coordSpecies = None
        self.presence_trees = {}  # Dictionary to hold BallTrees for each species
        self.training_trees = {}  # Dictionary to hold BallTrees for each species
        self.bltrees = {}
        self.species_shape = None



        path = os.path.join(raw_data_dir, 'Sandflies/speciesshape/species.shp')
        if os.path.exists(path):
            self.species_shape = gpd.read_file(path)
            self.species_shape = self.species_shape.to_crs("EPSG:4326")

        path = os.path.join(data_dir, 'trainingWS.csv')
        if os.path.exists(path):
            self.data = pd.read_csv(path)


        if self.data is not None and not self.data.empty:
            # remove unknown_unknown_unknown species
            self.data = self.data[self.data["dst"] > 0]
            self.data = self.data[self.data["species"] != "unknown_unknown_unknown"]
            temp = self.data[["species", "spec_code"]].drop_duplicates().sort_values(by="spec_code").set_index("spec_code")
            self.species = {code: species for code, species in temp["species"].items()}
            self.species_count = len(self.species)

            # -----------------------------------------------------------------------------------------
            # Make ball tree based on the both training dataset and Eduardo
            path = os.path.join(raw_data_dir, 'Sandflies/Eduardo/spatial_presence.csv')
            spatial_presence = pd.read_csv(path)
            records = []
            t1 = self.data[["lat", "lon", "spec_code", "species", "region_code"]]
            for row in t1.itertuples():
                t2 = spatial_presence[spatial_presence["mappingcode"] == row.region_code]
                if t2.shape[0] > 0:
                    if (row.species in t2.columns) and (t2[[row.species]].values[0] > 0):
                        records.append([row.lat, row.lon, row.spec_code, row.species, 1])
            temp = pd.DataFrame(records, columns=["lat", "lon", "spec_code", "species", "presence"])
            self.training_trees = self._build_ball_tree(temp)

            # -----------------------------------------------------------------------------------------
            # Make ball tree based on the Eduardo maps
            records = []
            for index, row in spatial_presence.iterrows():
                for spec_code, species in self.species.items():
                    if (species in spatial_presence.columns) and (row[species] > 0):
                        records.append([row.latitude, row.longitude, spec_code, species, 1])
            temp = pd.DataFrame(records, columns=["lat", "lon", "spec_code", "species", "presence"])
            self.presence_trees = self._build_ball_tree(temp)

    def get_species(self):
        return self.species

    # % it returns the number of all possible species
    def get_species_count(self):
        return self.species_count

    def get_species_index(self):
        return self.species_index

    def _build_ball_tree(self, data):
        bal_trees = {}
        coordinate_species = data[["lat", "lon", "spec_code"]]
        coordinate_species = coordinate_species.groupby(["lat", "lon", "spec_code"]).size().reset_index(name='counts')

        #pivot the data to have the specious as columns
        coordinate_species_pivot = coordinate_species.pivot(index=["lat", "lon"], columns="spec_code", values="counts").reset_index()
        coordinate_species_pivot = coordinate_species_pivot.fillna(0)

        species_codes = coordinate_species_pivot.columns[2:]
        for spec_code in species_codes:
            idx = coordinate_species_pivot[spec_code] > 0
            coordinates = coordinate_species_pivot[["lat", "lon"]][idx].values
            coordinates = np.deg2rad(coordinates)
            bal_trees[spec_code] = BallTree(coordinates, metric='haversine')

        return bal_trees

    def _ball_tree_query(self, coordinates, spec_codes=None):

        result = np.zeros((len(coordinates), len(spec_codes)))
        for idx, spec_code in enumerate(spec_codes):
            distances, _ = self.bltrees[spec_code].query(coordinates, k=1, return_distance=True)
            result[:, idx] = distances.ravel()
        return result

    def _ball_tree_query_radius(self, coordinates, r = 1):
        result = {}
        for spec_code in range(self.species_count):
            result[spec_code] = self.bltrees[spec_code].query_radius(coordinates, r=r, return_distance=True)
        return result

    def pruning(self, data: pd.DataFrame, max_dist=500, type="training"):

        if type == "training":
            self.bltrees = self.training_trees
        else:
            self.bltrees = self.presence_trees

        data = data.sort_values(by=["date", "lat", "lon", "spec_code"])
        coordinates = data[["lat", "lon"]].copy(deep=True)
        coordinates.drop_duplicates(subset=["lat", "lon"], inplace=True)
        coordinates = coordinates.values
        coordinates = np.deg2rad(coordinates)

        # int this line we found the species which exist in data
        # in some cases species can ve a non-sequence array like [0,1,3]
        spec_codes_desired = data["spec_code"].unique()
        # in some case we used 99 species code for Max,
        # then we removed the spec_code > self.species_count
        spec_codes_desired = spec_codes_desired[spec_codes_desired <= self.species_count]

        max_distance_rad = max_dist / 6371.0

        # find the nearest distance for each coordinate and species, rows are coordinates and columns are species
        nearest_distance = self._ball_tree_query(coordinates, spec_codes=spec_codes_desired)

        # for each cell in the nearest_distance, if the distance is less than max_distance_rad, then it is 0,
        # otherwise (value - max_distance_rad) * 6371.0 / max_dist
        nearest_distance = (nearest_distance - max_distance_rad) * 6371.0 / max_dist
        nearest_distance = np.maximum(nearest_distance, 0)

        # add a new column for the max, as an optimistic approach
        # in the new version, we do not need to add a new column for the max
        # nearest_distance = np.c_[nearest_distance, np.ones(len(nearest_distance))]

        # apply sigmoid function to the normalized distance
        ## distance_adjustment_factor = 2 / (1 + np.exp(d))

        distance_adjustment_factor = 2 / (1 + np.exp(nearest_distance.ravel()))

        pred = data["pred"].values
        pred[pred <= 0] = 0
        pred = pred * distance_adjustment_factor
        pred = pred.round(4).astype('float32')

        pred = pred.reshape(-1, len(spec_codes_desired))
        # get the maximum value of each row except the last column and assign it to the last column
        #pred[:, -1] = np.max(pred[:, :-1], axis=1)
        pred = pred.flatten()
        data["pred"] = pred

        return data

    def pruning_new(self, data: pd.DataFrame):
        gdf = gpd.GeoDataFrame(
            data,
            geometry=[Point(xy) for xy in zip(data.lon, data.lat)],
            crs="EPSG:4326"
        )
        results = []
        for sp in gdf['spec_code'].unique():

            gdf_sp = gdf[gdf['spec_code'] == sp].copy()
            poly_sp = self.species_shape[self.species_shape['spec_code'] == sp].copy()

            joined = gpd.sjoin(gdf_sp, poly_sp[['geometry']], how='left', predicate='within')
            joined['pred'] = joined.apply(
                lambda row: row['pred'] if (pd.notnull(row['index_right']) or pd.isnull(row['pred'])) else 0,
                axis=1
            )
            joined = joined.drop(columns=['index_right'])
            results.append(joined)

        result_gdf = pd.concat(results, ignore_index=True)
        result_df = pd.DataFrame(result_gdf.drop(columns=['geometry']))

        return result_df

    def save(self):
        attributes = {
            'species': self.species,
            "species_shape": self.species_shape,
            'species_count': self.species_count,
            'species_index': self.species_index,
            'coordSpecies': self.coordSpecies,
            'presences_trees': self.presence_trees,
            'training_trees': self.training_trees
        }
        path = os.path.join(self._model_dir, self.__name + ".joblib")
        joblib.dump(attributes, path)

    def load(self):
        path = os.path.join(self._model_dir, self.__name + ".joblib")
        attributes = joblib.load(path)
        self.species = attributes['species']
        self.species_shape = attributes['species_shape']
        self.species_count = attributes['species_count']
        self.species_index = attributes['species_index']
        self.coordSpecies = attributes['coordSpecies']
        self.presence_trees = attributes['presences_trees']
        self.training_trees = attributes['training_trees']
        return joblib.load(path)

    def set_model_directory_path(self, root):
        self._model_dir = root
        return self
