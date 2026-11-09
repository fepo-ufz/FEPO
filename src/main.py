import os
import dotenv
import logging
from pathlib import Path
from fepo.FEPO import FEPO
from fepo.FeSpecies import FeSpecies

seed = 321
logger = logging.getLogger("FEPO MAIN")

if __name__ == "__main__":

    dotenv.load_dotenv()

    root_directory = os.getenv("WORKING_DIR")
    if not root_directory :
        root_directory = Path(__file__).resolve().parents[1]

    raw_data_directory = os.getenv("RAW_DATA_DIR")
    artifact_directory = os.getenv("ARTIFACT_DIR")
    train_data_directory = os.path.join(root_directory, os.getenv("TRAIN_DATA_DIR"))
    model_directory = os.path.join(root_directory, os.getenv("MODEL_DIR"))

    fepo = FEPO().set_seed(seed).set_data_directory_path(train_data_directory)
    fepo = fepo.set_model_directory_path(model_directory).set_artifacts_directory_path(artifact_directory)

    fepo.set_verbose(1)
    fepo.set_rare_thresholds([0.01, 0.50]) # for a real scenario, please mark this line as a comment
    fepo.tumas_basic_models()
    fepo.tumas_meta_model()

    species = FeSpecies(raw_data_dir=raw_data_directory, data_dir=train_data_directory).set_model_directory_path(model_directory)
    species.save()
