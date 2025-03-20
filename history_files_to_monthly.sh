#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_directory> <case_name> <output_directory>"
    exit 1
fi

# Assign input arguments to variables
input_dir="$1"
case_name="$2"
output_dir="$3"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Find all files matching the pattern and extract unique year-month combinations
declare -A month_map
for file_path in "$input_dir"/${case_name}.cam.h1.*.nc; do
    # Extract the file name
    file_name=$(basename "$file_path")
    # Parse year and month from the file name
    year_month=$(echo "$file_name" | awk -F. '{print $4"-"$5}')
    # Store unique year-month
    month_map["$year_month"]=1
done

# Iterate over the unique year-month combinations and process files
for year_month in "${!month_map[@]}"; do
    # Extract year and month
    year="${year_month:0:4}"
    month="${year_month:5:2}"

    # Construct the output file name
    output_file="${output_dir}/${case_name}.cam.h1.${year}-${month}.monthly_avg.nc"

    # Construct the pattern to match the files
    pattern="${case_name}.cam.h1.${year}-${month}-*.nc"

    # Perform the monthly mean operation
    echo "Processing ${year}-${month}..."
    cdo monmean -mergetime "${input_dir}/${pattern}" "$output_file"
done

echo "Monthly averaging completed successfully."
