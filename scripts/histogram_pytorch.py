from pathlib import Path
import xarray as xr

import torch
from torch.utils.data import DataLoader, IterableDataset

class DaskArrayDataset(IterableDataset):
    def __init__(self, dask_array):
        self.dask_array = dask_array

    def __iter__(self):
        # A Dask array is made of chunks. This loads one chunk at a time.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # In main process
            chunk_indices = range(self.dask_array.npartitions)
        else:  # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            chunk_indices = range(worker_id, self.dask_array.npartitions, num_workers)
        
        for idx in chunk_indices:
            # Dask will load the chunk from disk and convert it to a numpy array
            chunk = self.dask_array.to_delayed()[idx].compute()
            yield chunk


if __name__ == "__main__":
    loc = Path("/Users/brianpm/Dropbox/Data")
    fil = "tas_Amon_HadGEM3-GC31-LL_historical_r3i1p1f3_gn_195001-201412.nc"

    # Open a single large NetCDF file or a collection of them
    ds = xr.open_dataset(
        loc/fil,
        engine="netcdf4"
    )

    # Select the variable of interest, flatten it, and convert to a Dask array
    data_variable = ds["tas"].data.flatten()


    # Initialize dataset and dataloader
    # dataset = DaskArrayDataset(data_variable)
    # data_loader = DataLoader(dataset, batch_size=None, num_workers=4)

    # data_loader = DataLoader(data_variable, batch_size=None, num_workers=4)
    # print(data_loader)

    # Check for GPU availability
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define histogram parameters
    num_bins = 256
    hist_min = float(data_variable.min().item())
    hist_max = float(data_variable.max().item())
    print(f"N={num_bins}, from {hist_min} to {hist_max}")
    data_tensor = torch.from_numpy(data_variable).to(device, non_blocking=True)
    # Calculate the histogram for the current batch
    # data_histogram = torch.histogram(
    #     input=data_tensor,
    #     bins=int(200),
    #     range=(hist_min, hist_max)
    # )
    bin_edges = torch.linspace(hist_min, hist_max, steps=num_bins + 1)
    data_histogram = torch.histogram(input=data_tensor, bins=bin_edges)

    print("Done with histogram")
    print(data_histogram)

    # Move the final histogram to the CPU for any further processing or plotting
    final_histogram_cpu = data_histogram.hist.cpu()
    # print(final_histogram_cpu)