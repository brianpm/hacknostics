#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import sys
import subprocess
from pathlib import Path

# ncrcat -4 -v FSNT F1850JJB_c201_CTL.cam.h0.* /project/amp/brianpm/F1850JJB_c201_CTL.cam.h0.ncrcat.FSNT.nc

# passing list of variables
# https://stackoverflow.com/questions/15753701/argparse-option-for-passing-a-list-as-option

# globbing for files from argument
# https://stackoverflow.com/questions/12117467/how-to-pass-files-to-script-with-glob-and-specific-path-with-error-if-path-does

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a timeseries file from a list of files."
    )
    parser.add_argument(
        "--variables", nargs="+", help="one or more variables to process."
    )
    parser.add_argument(
        "files",
        metavar="path",
        type=str,
        help="Path to files to be merged; enclose in quotes, accepts * as wildcard for directories or filenames",
    )
    parser.add_argument("output", metavar="path", type=str, help="output directory")
    args = parser.parse_args()
    # what if they pass a comma-separated list of variables?
    vlist = args.variables
    print(f"The variables list is {type(vlist)} with length {len(vlist)}")
    if isinstance(vlist, str):
        print("[vlist is string]")
        if "," in args.variables:
            print("[vlist has comma -- SPLIT]")
            vlist = vlist.split(",")
    elif isinstance(vlist, list):
        vlist = [i.split(",") for i in vlist]
        vlist = [item for sublist in vlist for item in sublist]
    else:
        print("[vlist out of luck.]")
    print(vlist)
    files = sorted(glob.glob(args.files))
    if not files:
        print("File does not exist: " + args.files, file=sys.stderr)
    else:
        print(f"Input seems to exist, pass {args.files} to ncrcat")
    # for file in files:
    #     print('File exists: ' + file)

    # if more than one variable, then we need to have a way to construct the output
    # I guess we just do it even for one variable; user has to deal with renaming afterward.
    out_loc = Path(args.output)
    assert out_loc.is_dir()
    # try to guess at the file name using the input file
    first_in = Path(files[0])
    case_spec = first_in.stem  # this will be the filename without suffix

    for v in vlist:
        out_file = out_loc / f"{case_spec}.ncrcat.{v}.nc"
        print(f"Writing to: {out_file}")
        cmd = ["ncrcat", "-4", "-v", v] + files + ["-o", out_file]
        subprocess.run(cmd)
