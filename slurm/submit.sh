#!/bin/bash
#SBATCH --job-name=FepoWrapper
#SBATCH --array=0-10
#SBATCH --output=./Logs/logs/%x-%A-%a.out
#SBATCH --error=./Logs/err/%x-%A-%a.err
#SBATCH -t 0-23:00:00
#SBATCH --mem-per-cpu=15G


echo "------------------------------------------------------------"
echo "$(date): Job $SLURM_ARRAY_TASK_ID started."
echo "------------------------------------------------------------"

# ------------------------------------------------------------------------------
# Load Python environment
# ------------------------------------------------------------------------------
echo "$(date): Loading Conda environment..."
module purge
module load Conda/24.3.0
source activate envFepo

if [ $? -ne 0 ]; then
    echo "$(date): ERROR - Failed to activate Conda environment."
    exit 1
fi

# ------------------------------------------------------------------------------
# Paths (Update as needed)
# ------------------------------------------------------------------------------

echo "SLURM_SUBMIT_DIR = $SLURM_SUBMIT_DIR"

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="$PROJECT_ROOT/slurm"
SCRIPT_PATH="$SCRIPT_DIR/Wrapper.py"

echo "Script directory : $SCRIPT_DIR"
echo "Project root     : $PROJECT_ROOT"
cd "$PROJECT_ROOT" || {
    echo "ERROR: Cannot cd to project root"
    exit 1
}


NC_MORPH_DIR="/gpfs1/data/climos/FEPO/ProcessedData/Morphology"
NC_METEO_DIR="/gpfs1/data/climos/FEPO/ProcessedData/Meteorology"
NC_HYDRO_DIR="/gpfs1/data/climos/FEPO/ProcessedData/Hydrology"
REGION="39.50 -0.95 39.00 -0.90" # <=== set region here (top_left_lat top_left_lon bottom_right_lat bottom_right_lon)
PERIOD="2022-07-01 2022-07-31"   # <=== set period here (start_date end_date)

# CLIMOS Domain, in the real sceneries
# REGION="54.00 -10.00 35.00 45.00"
# PERIOD="2022-01-01 2022-12-31"

# ------------------------------------------------------------------------------
# Set default SLURM array variables for testing outside of SLURM
# ------------------------------------------------------------------------------

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
    SLURM_ARRAY_TASK_COUNT=100
fi

echo "$(date): FEPO processing started."

# ------------------------------------------------------------------------------
# Run the Python FEPO wrapper
# ------------------------------------------------------------------------------
python "$SCRIPT_PATH" \
    --nc_morph_dir "$NC_MORPH_DIR" \
    --nc_meteo_dir "$NC_METEO_DIR" \
    --nc_hydro_dir "$NC_HYDRO_DIR" \
    --region $REGION \
    --period $PERIOD \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    --num_jobs "$SLURM_ARRAY_TASK_COUNT"
status=$?

if [ $status -ne 0 ]; then
    echo "$(date): ERROR - FEPO script failed with exit code $status."
    exit $status
fi

echo "$(date): FEPO completed successfully."
echo "Job number $SLURM_ARRAY_TASK_ID completed."

# ------------------------------------------------------------------------------
# Report MaxRSS for this array task
# ------------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "$(date): SLURM memory usage report"

JOBID_FULL="$SLURM_JOB_ID"
sleep 10

MAXRSS=$(sacct -j "$JOBID_FULL" \
    --format=JobID,MaxRSS \
    --noheader \
    | grep '+' \
    | awk '{print $2}')

if [ -n "$MAXRSS" ]; then
    echo "$(date): MaxRSS = $MAXRSS"
else
    echo "$(date): MaxRSS not available"
fi

echo "------------------------------------------------------------"