import argparse
import pandas as pd
import sys
import os
import dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from slurm.Extracting import Extracting
from src.fepo.FEPO import FEPO
from src.fepo.FeSpecies import FeSpecies

def main():

    dotenv.load_dotenv()
    model_dir = os.getenv("MODEL_DIR")
    artifact_dir= os.getenv("ARTIFACT_DIR")
    args = parser.parse_args()

    if exists_result_file(args.job_id, artifact_dir):
        print(f"Result file for job_id {args.job_id} already exists. Skipping computation.")
        return

    ext = Extracting(
        nc_morphology_dir=args.nc_morph_dir,
        nc_meteo_dir=args.nc_meteo_dir,
        nc_hydro_dir=args.nc_hydro_dir,
        region=tuple(args.region),
        date_period=tuple(args.period),
        spatial_resolution=args.spatial_res,
        temporal_resolution=args.temporal_res)

    ext.set_date_name("sdate") # It is important because the training dataset for basic models uses "sdate"
    ext.set_batch_size(500_000)
    features = ext.get_features_data(args.job_id, args.num_jobs)
    fepo = FEPO().set_model_directory_path(model_dir)
    fepo.set_rare_thresholds([0.01, 0.50]) ## <==============
    predictions = fepo.predict(features, species_codes=args.species)

    pruning = FeSpecies().set_model_directory_path(model_dir)
    pruning.load()
    predictions = pruning.pruning_new(predictions)

    os.makedirs(artifact_dir, exist_ok=True)
    output_file = get_result_file_path(args.job_id, artifact_dir)
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def get_result_file_path(job_id:int, artifact_dir:str) -> str:
    job_id_str = f"{job_id:05d}"
    output_file = os.path.join(artifact_dir, f"predictions_{job_id_str}.csv")
    return output_file

def exists_result_file(job_id:int, artifact_dir:str) -> bool:
    output_file = get_result_file_path(job_id, artifact_dir)
    return os.path.isfile(output_file)

if __name__ =="__main__":
    """
     ---region: Geographical region as top_peft_lat, top_left_lon, bottom_right_lat, bottom_right_lon for example [41.0 -8.0 39.0 -6.0]
     ---date_period: Date period as start_date end_date (YYYY-MM-DD) for example [2020-01-01 2020-12-31]
    """

    parser = argparse.ArgumentParser(description="Wrapper for running the FEPO model in parallel in a SLURM environment.")
    parser.add_argument('--nc_morph_dir', type=str, required=True, help='Path to the morphology NetCDF files directory')
    parser.add_argument('--nc_meteo_dir', type=str, required=True, help='Path to the meteorological NetCDF files directory')
    parser.add_argument('--nc_hydro_dir', type=str, required=True, help='Path to the hydrological NetCDF files directory')
    parser.add_argument('--region', type=float, nargs=4, required=True, help='Geographical region as min_lat min_lon max_lat max_lon')
    parser.add_argument('--period', type=str, nargs=2, required=True, help='Date period as start_date end_date (YYYY-MM-DD)')
    parser.add_argument('--spatial_res', type=float, default=0.015625, help='Spatial resolution in degrees')
    parser.add_argument('--temporal_res', type=str, default="1D", help='Temporal resolution (e.g., "1D" for daily)')
    parser.add_argument('--species', type=int, nargs='+', default=[0,1,2,3], help='List of species codes to predict')
    parser.add_argument('--job_id', type=int, required=True, help='SLURM job ID for parallel processing')
    parser.add_argument('--num_jobs', type=int, required=True, help='Total number of SLURM jobs for parallel processing')
    main()
