import os
import subprocess

def compress_netcdf_file(file_path):
    # Define the output file path
    compressed_file_path = file_path[:-3] + '_compressed.nc'
    
    # Command to compress using ncks with specified options
    command = [
        'ncks',
        '-4', 
        '--qnt', 'dfl=4', 
        '--cmp', 'shf|zlib,5', 
        file_path, 
        compressed_file_path
    ]
    
    try:
        # Run the compression command
        subprocess.run(command, check=True)
        # If compression is successful, remove the original file
        os.remove(file_path)
        # Optionally rename the compressed file to match the original
        os.rename(compressed_file_path, file_path)
        print(f'Compressed and replaced {file_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error compressing {file_path}: {e}')
    except Exception as e:
        print(f'An unexpected error occurred while processing {file_path}: {e}')

def find_and_compress_netcdf_files(start_directory):
    for root, dirs, files in os.walk(start_directory):
        for file in files:
            if file.endswith('.nc'):  # Check for netCDF files
                file_path = os.path.join(root, file)
                compress_netcdf_file(file_path)

# Path to your small test directory
directory_path = '/Volumes/Alsakan'
find_and_compress_netcdf_files(directory_path)
