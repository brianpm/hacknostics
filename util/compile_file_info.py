import argparse
from pathlib import Path
import xarray as xr


def _writeline(ofil, lines):
    outF = open(ofil, "a")
    if isinstance(lines, str):
        outF.write(lines)
    else:
        for l in lines:
            outF.write(l)
            outF.write("\n")
    outF.close()


def _main(pth, out):
    # list of files:
    lof = pth.glob('**/*')
    for f in lof:
        if f.suffix == '.nc':
            try:
                ds = xr.open_dataset(f,decode_cf=False,decode_times=False)
            except:
                continue
            send_to_file = [str(f)]
            [send_to_file.append(f"\t {j}") for j in ds.data_vars]
            _writeline(out, send_to_file)



if __name__ == "__main__":
    # treat the argument as the root directory to start checking.
    parser = argparse.ArgumentParser(
        description="Check for netCDF files under a given directory."
    )
    parser.add_argument(
        "start",
        metavar="path",
        type=str,
        help="Path to start directory; enclose in quotes, accepts * as wildcard for directories or filenames",
    )
    parser.add_argument("output", metavar="path", type=str, help="output text file")
    args = parser.parse_args()
    # create the output file:
    start_dir = Path(args.start)
    # make sure the starting point exists
    if not start_dir.exists():
        raise(IOError("Directory doesn't exist."))
    output_fil = Path(args.output)
    if not output_fil.is_file():
        output_fil.touch()
    _main(start_dir, output_fil)
    print("Finished.")