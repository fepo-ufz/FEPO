# FEPO: Sandfly Extreme Population Prediction model

FEPO predicts daily densities of phlebotomine sandflies from meteorological, hydrological and morphological predictors with a stacked ensemble of gradient boosted trees trained on light-trap catches.

FEPO was developed by Majid Soheili at the Helmholtz Centre for Environmental Research - UFZ in the
CLIMOS project (EU Horizon Europe, grant agreement 101057690).

- Article: Soheili et al. (2025), FEPO: a machine learning ensemble approach for predicting extreme
  phlebotomine sand fly abundance and leishmaniasis-risk hotspots across Europe,
  <https://doi.org/10.21203/rs.3.rs-7852217/v1>
- Output for the CLIMOS domain, 1999-2010: <https://doi.org/10.5281/zenodo.24081829>
- Training tables are produced with PreFEPO: <https://github.com/fepo-ufz/PreFEPO>

## Release 1.0.0

Release 1.0.0 is the version used for the CLIMOS runs (commit 3a991cd of 3 January 2026 for FEPO,
commit 9f04ff7 of 23 December 2025 for PreFEPO, both in the internal UFZ repositories). It is published
as it was run, apart from the files added for the release (licence, authors, citation) and the
installation of smogn from its public repository. The changes are listed in `CHANGELOG.md`.

The trap data used for training belong to the providers of the field surveys and are not part of this
repository. They can be requested from the providers listed in the article.

---

# SandFlies Extreme Population Prediction model (FEPO)
In this repository, we present the SandFlies Extreme Population Prediction model (FEPO), which aims to predict extreme population events of sandflies based on various environmental factors.
## STEP 0: Set up the environment
 - A script named `Setup.sh` is provided in the repository to automate the environment setup process.
 - You can run this script to create the FEPO environment with all necessary packages installed.
 - After cloning the repository, execute the following command in your terminal:
    ```bash
    chmod +x ./Setup.sh
    ./Setup.sh
    ```
## STEP 1: Training Dataset
- The training data provided by PreFEPO project.
- The training data is stored in the `./resources/training`.
- The training data included:
    - Sandflies population data. (locations, dates, species, density)
    - Morphological data (slope, aspect, elevation, land cover, sand, clay)
    - Meteorological data (temperature, precipitation, potential evaporation)
    - Hydrological data (soil moisture in 6 depths)
    
    | lat   | lon    | sdate      | species | region | f1  | f2  | f3  | ... | fn   | density |
    |-------|--------|------------|---------|--------|-----|-----|-----|-----|------|---------|
    | 30.01 | 5.12   | 2020-07-01 | 1       | 1      | 0.5 | 0.2 | 0.8 |     | 0.12 | 6       |
    | 51.30 | 5.50   | 2020-08-15 | 2       | 1      | 0.1 | 0.8 | 0.5 |     | 0.75 | 3       |
    | 38.50 | -9.031 | 2011-06-23 | 1       | 2      | 0.3 | 0.2 | 0.1 |     | 0.35 | 2.5     |    

- In the training table, the identity columns include:	(lat, lon, sdate, species, and region)
- The target column is density, which should be the logarithm of the sandfly density.

---
## STEP 2: Set Feature SubSet
- The feature subset should be defined in the DataHandler class at `src/fepo/DataHandler.py`
  1. `fss_data` refers to feature subset selection.
  2. `index_columns` is a part of `fss_data` and refers to identity features.
  3. `cat_features` is a part of `fss_data` and refers to categorical features.
  4. `irrelevant_columns` is a part of `fss_data` and refers to irrelevant columns.
     - these features would be useful for downstream tracking but won't be used in training.
  5. `target` is a columns refers to density of sandflies.
  ``` python
    __index_columns = ["lat", "lon", "sdate", "region_code", "expid"]
    __target = "dst"
    __cat_features = ["spec_code", "lc"]
    __irrelevant_columns = ["ds", 'species'] 
    __fss_data = ["spec_code", "tavg_365", "clay", "elevation", "aspect", "pet_365", ... ,"slope"]
    ```
---
## STEP 3: Set Global Address
Set addresses in the.env file placed at the root directory
   - RAW_DATA_DIR=/Volumes/Researches/CHS/CLIMOS/FeData/raw/
   - TRAIN_DATA_DIR=/absolute/path/to/training/data/
   - MODEL_DIR=/absolute/path/to/store/models/
   - ARTIFACT_DIR=/absolute/path/to/store/artifacts/
---
## STEP 4: training base models
1. For training and making the required models, you can use the `main.py` script located in the `src` directory.
2. by running the script, the basic models and metamodel will be trained and stored in the specified MODEL_DIR.
3. these models will be used for making predictions in the FEPO model.
    ``` bash 
    python src/main.py
    ```
---
## STEP 5: Running FEPO model on remote HPC cluster
- To run the FEPO model on a remote HPC cluster,
  you can use the bash script `submit.sh` located in the `slurm` directory.
- Set the required parameters in the `submit.sh` script,
  such as time, memory, and number of array job.
- Set the required path for extracting features and storing results.
  * NC_MORPH_DIR="/gpfs1/data/climos/FEPO/ProcessedData/Morphology"
  *   NC_METEO_DIR="/gpfs1/data/climos/FEPO/ProcessedData/Meteorology"
  *   NC_HYDRO_DIR="/gpfs1/data/climos/FEPO/ProcessedData/Hydrology"
  *   MODEL_DIR="$PROJECT_ROOT/resources/models"
  *   OUT_DIR="/gpfs1/data/climos/fepo_artifacts"
- Set Spatial and Temporal bounds in the `submit.sh` script.
  * REGION="top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon"
    * Example: REGION="39.50 -0.95 39.00 -0.90"
  * PERIOD="start_date,end_date"
    * Example: PERIOD="2023-06-01,2023-09-30
- Set BatchSize in the `Wrapper.py` script.
  * ext.set_batch_size(500_000)
* Set the rare thresholds in the `Wrapper.py` script.
  * in the real scenario, it is necessary to set.

- **CLIMOS Domain**:  
    * REGION="54.00 -10.00 35.00 45.00"
    * PERIOD="2022-01-01 2022-12-31"

---
## STEP 6: Merging the intermediate results 
- After completion of all array jobs, you will have multiple intermediate result files in the output directory.
- For merging these intermediate result files into a single NetCDF file, you can use the `Merger.py` script located in the `slurm` directory.
- Set the required parameters in the `Merger.py` script, such as input directory, output file path
  * input_dir = "/gpfs1/data/climos/fepo_artifacts"
  * output_dir = "/gpfs1/data/climos/fepo_artifacts"

---

## Citation

If you use FEPO, please cite the article above and this software (see `CITATION.cff`; the DOI of
each release is on Zenodo).

## Licence

LGPL-3.0-or-later (c) 2025-2026 the FEPO developers, Helmholtz Centre for Environmental Research - UFZ.
See `LICENSE.md`, `COPYING` and `COPYING.LESSER`.
