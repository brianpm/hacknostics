#!/bin/bash -l
### Job name
#PBS -N dy1_zarr
#PBS -A P93300313
#PBS -l select=1:ncpus=4:mem=256GB
#PBS -l walltime=01:30:00
#PBS -q casper
#PBS -j oe

# Set error handling
set -e
set -u
module purge # Start with a clean environment
module load conda

conda activate p12

# Set working directory to script location
cd /glade/u/home/brianpm/Code/hacknostics/healpix

# Print environment info
echo "Job started at $(date)"
echo "Working directory: $PWD"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Define the variable for the resubmit file
export RESUBMIT_FILE="/glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND1/resubmit.txt"
# export RESUBMIT_FILE="/glade/campaign/cgd/cas/brianpm/hack25/mpas_DYAMOND1/resubmit.txt"

# Run the script - Make sure python executable from conda env is used
python convert_mpas_dyamond1_to_healpix_zarr.py

# Check the resubmit status
if [[ -f "$RESUBMIT_FILE" ]]; then
    RESUBMIT_STATUS=$(<"$RESUBMIT_FILE")
    if [[ $RESUBMIT_STATUS == "TRUE" ]]; then
        echo "More files to process. Resubmitting."
        qsub "$0"
    else
        echo "All files processed. Exiting."
    fi
else
    echo "$RESUBMIT_FILE does not exist. Exiting."
fi

echo "Job finished at $(date)"
