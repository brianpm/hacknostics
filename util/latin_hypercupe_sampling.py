"""
Input a YAML file following this scheme:

parameter0:
  range: [min, max]
parameter1:
  range: [0, 1]
parameter2:
  range: [-10, 10]
parameter3:
  range: [100, 1000]
"""
import yaml
import numpy as np
from scipy.stats import qmc
import argparse
from netCDF4 import Dataset

def load_parameters(yaml_file):
    """Load parameter definitions from a YAML file."""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def generate_lhs_samples(parameters, num_samples):
    """Generate Latin Hypercube Samples for the given parameters."""
    sampler = qmc.LatinHypercube(d=len(parameters))
    samples = sampler.random(n=num_samples)
    
    scaled_samples = []
    for i, (param, values) in enumerate(parameters.items()):
        min_val, max_val = values['range']
        scaled = qmc.scale(samples[:, i], min_val, max_val)
        scaled_samples.append(scaled)
    
    return np.array(scaled_samples).T

def save_to_netcdf(samples, params, output_file):
    """Save the generated samples to a NetCDF file."""
    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        # Create dimensions
        nc.createDimension('sample', samples.shape[0])
        nc.createDimension('parameter', samples.shape[1])
        
        # Create variables
        sample_var = nc.createVariable('sample', 'i4', ('sample',))
        sample_var[:] = np.arange(samples.shape[0])
        
        param_var = nc.createVariable('parameter', 'S1', ('parameter',))
        param_var[:] = np.array(list(params.keys()), dtype='S')
        
        data_var = nc.createVariable('values', 'f8', ('sample', 'parameter'))
        data_var[:] = samples
        
        # Add attributes
        nc.description = 'Parameter sets generated using Latin Hypercube Sampling'
        for i, (param, values) in enumerate(params.items()):
            data_var.setncattr(f'{param}_range', str(values['range']))

def main(yaml_file, num_samples, output_file):
    # Load parameters from YAML file
    params = load_parameters(yaml_file)

    # Generate LHS samples
    samples = generate_lhs_samples(params, num_samples)

    # Save to NetCDF file
    save_to_netcdf(samples, params, output_file)

    print(f"Generated {num_samples} parameter sets and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parameter sets using Latin Hypercube Sampling")
    parser.add_argument("yaml_file", help="Input YAML file with parameter definitions")
    parser.add_argument("num_samples", type=int, help="Number of parameter sets to generate")
    parser.add_argument("output_file", help="Output NetCDF file to save the generated parameter sets")
    args = parser.parse_args()

    main(args.yaml_file, args.num_samples, args.output_file)