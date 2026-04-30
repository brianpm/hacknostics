#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
from fpdf import FPDF, XPos, YPos

class NetCDFToPDF(FPDF):
    def __init__(self, filename=""):
        super().__init__()
        self.filename = filename
        self.set_margins(15, 22, 15)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_fill_color(28, 40, 65)
        self.rect(0, 0, self.w, 18, style="F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.set_y(2)
        self.cell(0, 14, "NetCDF Metadata Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_text_color(33, 37, 41)

    def footer(self):
        self.set_y(-15)
        self.set_text_color(120, 120, 130)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_text_color(33, 37, 41)

    def add_section_header(self, title):
        """Renders a filled band for section titles."""
        self.set_fill_color(70, 90, 120)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(3)
        self.set_text_color(33, 37, 41)

    def add_summary_table(self, metadata):
        """Renders a three-column summary table."""
        n_dims = len(metadata["dimensions"])
        n_vars = len(metadata["variables"])
        n_global = len(metadata["global_attributes"])

        col_w = self.epw / 3

        self.set_fill_color(28, 40, 65)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)

        self.cell(col_w, 8, "Dimensions", fill=True, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w, 8, "Variables", fill=True, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w, 8, "Global Attributes", fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_fill_color(245, 246, 248)
        self.set_text_color(33, 37, 41)
        self.set_font("Helvetica", "", 11)

        self.cell(col_w, 10, str(n_dims), fill=True, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w, 10, str(n_vars), fill=True, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(col_w, 10, str(n_global), fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_draw_color(210, 215, 220)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def add_variable_section(self, var_name, dimensions, attributes, is_coord, index):
        """Renders a variable card with name and attributes."""
        self.set_draw_color(210, 215, 220)

        if is_coord:
            self.set_fill_color(46, 134, 171)
        else:
            self.set_fill_color(55, 65, 81)

        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.set_x(self.l_margin)
        dim_str = f"{var_name}{dimensions}"
        self.multi_cell(self.epw, 8, dim_str, fill=True)

        if index % 2 == 0:
            self.set_fill_color(248, 249, 250)
        else:
            self.set_fill_color(255, 255, 255)

        self.set_text_color(60, 60, 70)
        self.set_font("Helvetica", "", 9)

        if not attributes:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 110)
            self.set_x(self.l_margin)
            self.cell(self.epw, 6, "  (no attributes)", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            for attr_name, attr_val in attributes.items():
                attr_text = f"  {attr_name}: {attr_val}"
                self.set_x(self.l_margin)
                self.multi_cell(self.epw, 6, attr_text, fill=True)

        self.set_draw_color(210, 215, 220)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

def parse_ncdump(ncdump_path, file_path):
    """Parses ncdump -h output into dimensions, variables, and global attributes."""
    try:
        result = subprocess.run(
            [ncdump_path, "-h", file_path],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running ncdump: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: ncdump not found at {ncdump_path}")
        sys.exit(1)

    lines = result.stdout.splitlines()

    data = {
        "dimensions": {},
        "variables": [],
        "global_attributes": {}
    }

    state = "preamble"
    current_var = None
    dimensions = set()
    type_keywords = ["float", "double", "int", "int64", "uint", "uint64",
                     "short", "ushort", "byte", "ubyte", "char", "string"]

    for line in lines:
        line = line.strip()
        if not line or line in ("{", "}", "},"):
            continue

        if line == "dimensions:":
            state = "dimensions"
            continue
        elif line == "variables:":
            state = "variables"
            current_var = None
            continue
        elif line.startswith("// global attributes:"):
            state = "global_attrs"
            current_var = None
            continue
        elif line.startswith("//"):
            continue

        if state == "dimensions":
            if "=" in line and not line.startswith(":"):
                parts = line.split("=", 1)
                dim_name = parts[0].strip()
                size_str = parts[1].strip().rstrip(";").split("//")[0].strip()

                if size_str.startswith("UNLIMITED"):
                    size = None
                else:
                    try:
                        size = int(size_str)
                    except ValueError:
                        size = None

                data["dimensions"][dim_name] = size
                dimensions.add(dim_name)

        elif state == "variables":
            if any(line.startswith(t) for t in type_keywords) and "(" in line:
                try:
                    start_paren = line.find('(')
                    end_paren = line.find(')')
                    name_part = line[:start_paren].split()[-1]
                    dims_part = line[start_paren:end_paren+1]
                    is_coord = name_part in dimensions

                    current_var = {
                        "name": name_part,
                        "dims": dims_part,
                        "attributes": {},
                        "is_coord": is_coord
                    }
                    data["variables"].append(current_var)
                except Exception:
                    pass
            elif ":" in line and "=" in line and current_var is not None:
                parts = line.split("=", 1)
                key_part = parts[0].strip()
                val_part = parts[1].strip().rstrip(";").strip().strip('"')

                if ":" in key_part:
                    _, attr_key = key_part.split(":", 1)
                    current_var["attributes"][attr_key.strip()] = val_part

        elif state == "global_attrs":
            if line.startswith(":") and "=" in line:
                parts = line.split("=", 1)
                attr_name = parts[0].strip().lstrip(":")
                val_part = parts[1].strip().rstrip(";").strip().strip('"')
                data["global_attributes"][attr_name] = val_part

    return data

def main():
    parser = argparse.ArgumentParser(description="Convert NetCDF metadata to a clean PDF report.")
    parser.add_argument("input", help="Path to the input NetCDF file")
    parser.add_argument("-o", "--output", help="Path to the output PDF file", default="metadata_report.pdf")
    parser.add_argument("-n", "--ncdump", help="Path to ncdump executable",
                        default="/Users/brianpm/miniforge3/envs/p12/bin/ncdump")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print(f"Parsing {args.input}...")
    metadata = parse_ncdump(args.ncdump, args.input)

    print(f"Generating PDF: {args.output}...")
    display_name = os.path.basename(args.input)
    pdf = NetCDFToPDF(filename=display_name)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(28, 40, 65)
    pdf.cell(0, 9, f"File: {display_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(33, 37, 41)
    pdf.ln(3)

    pdf.add_summary_table(metadata)

    pdf.add_section_header("Global Attributes")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 70)
    pdf.set_fill_color(245, 246, 248)

    if metadata["global_attributes"]:
        for k, v in metadata["global_attributes"].items():
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 6, f"  {k}: {v}", fill=True)
    else:
        pdf.set_x(pdf.l_margin)
        pdf.cell(pdf.epw, 6, "  None", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(6)
    pdf.add_section_header("Variables")

    if not metadata["variables"]:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(33, 37, 41)
        pdf.cell(0, 10, "No variables found.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        for idx, var in enumerate(metadata["variables"]):
            pdf.add_variable_section(
                var["name"], var["dims"], var["attributes"],
                is_coord=var["is_coord"], index=idx
            )

    pdf.output(args.output)
    print("Done!")

if __name__ == "__main__":
    main()
