#!/usr/bin/env python
# coding: utf-8

#
# This shows an example of using python instead of a shell script.
# It acts as a command-line interface (CLI) by accepting arguments (using argparse).
# Not a lot of features are built in to this; it could be extended to have much more functionality.
# It takes a path or a simple pattern, looks for postscript files (.ps) in that location, and
# invokes the system's `ps2pdf` command to create pdf versions of the files. 
#
# Brian Medeiros, 8 June 2022
#
import argparse
import glob
import subprocess
from pathlib import Path

def main(infils, out_loc=None):
    # check if infils has a wildcard (*) --> indicates glob
    # put results into list of Path objects
    if "*" in infils:
        tmpfils = [Path(i) for i in glob.glob(infils)]
    else:
        tmpfils = [Path(infils)]
    print(f"Going to process {len(tmpfils)} possible locations.")
    finalfils = []
    for tf in tmpfils:
        if tf.is_dir():
            # descend into directory tree, or just take first level?
            print(f"Identified directory: {tf.name}. Will find all ps files in that directory.")
            tmpfils2 = list(tf.glob("*.ps"))
            if tmpfils2:
                finalfils += tmpfils2
        elif tf.is_file():
            finalfils.append(tf)
    if finalfils:
        print(f"Identified a total of {len(finalfils)} files to convert.")
    else:
        print("No files to process. Exiting.")
        return None
    for v in finalfils:
        suffix = v.suffix
        if suffix != ".ps":
            continue  # SKIP NON-PS FILES
        # vname = v.name # includes .ps
        vstem = v.stem # no suffix
        # check if an output location is specified, if not use each file's location
        if out_loc is not None:
            out_file = out_loc / f"{vstem}.pdf"
        else:
            out_file = v.parent / f"{vstem}.pdf"
        print(f"\t Writing to: {out_file}")
        cmd = ["ps2pdf", "-dEPSCrop", v, out_file]
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PS file(s) to PDF."
    )
    parser.add_argument(
        "--src",
        metavar="path",
        type=str,
        help="Path to files to be converted; enclose in quotes, accepts * as wildcard for directories or filenames",
    )
    args = parser.parse_args()
    main(args.src)