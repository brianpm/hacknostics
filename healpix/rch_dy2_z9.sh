#!/bin/bash
#PBS -N rechunker
#PBS -A P93300042
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:mem=256GB
#PBS -q casper
#PBS -j oe
#PBS -k eod

set -e  # Exit on any error

# Load required modules
module load conda
conda activate p12

# Change to working directory
cd /glade/u/home/brianpm/Code/hacknostics/healpix

# Run the python script
# Replace with your actual input/output paths and any other arguments
python simpler_rechunker_v1.py \
    /glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND2/15min/DYAMOND2_diag_15min_to_hp9.zarr \
    /glade/derecho/scratch/digital-earths-hackathon/mpas_DYAMOND2/15min/DYAMOND2_diag_PT15M_to_hp9.zarr \
    --time-chunk 12 \
    --cell-chunk 196608 \
    --fix-time \
    --time-start "2020-01-20 00:00:00" \
    --time-end "2020-03-01 00:00:00" \
    --time-freq 15min