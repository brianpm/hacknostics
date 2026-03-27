import re
import os
import argparse

def count_non_binary_floats(file_list, comment_char):
    # Regex for floating point numbers (including scientific notation)
    float_pattern = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.\d*(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+")
    
    per_file_counts = {}
    total_count = 0

    for file_path in file_list:
        if not os.path.isfile(file_path):
            print(f"⚠️  Skipping '{file_path}': Not a valid file.")
            continue
            
        file_instances = 0
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # 1. Strip everything after the comment character
                    # 2. Strip whitespace from what remains
                    clean_line = line.split(comment_char, 1)[0].strip()
                    
                    if not clean_line:
                        continue
                    
                    # Search only the 'active' part of the line
                    matches = float_pattern.findall(clean_line)
                    for match in matches:
                        try:
                            val = float(match)
                            # Exclude values that are identically 0.0 or 1.0
                            if val != 0.0 and val != 1.0 and val != -1.0 and val != 2.0:
                                file_instances += 1
                                #print (val)
                        except ValueError:
                            continue
            
            per_file_counts[file_path] = file_instances
            total_count += file_instances
            
        except Exception as e:
            print(f"❌ Error reading '{file_path}': {e}")

    return per_file_counts, total_count

def main():
    parser = argparse.ArgumentParser(
        description="Count non-0/1 floats, ignoring full-line and inline comments."
    )
    parser.add_argument("files", nargs="+", help="Files or wildcards to parse")
    parser.add_argument(
        "--comment", 
        default="!", 
        help="The character(s) used for comments (default: #)"
    )
    
    args = parser.parse_args()

    per_file, total = count_non_binary_floats(args.files, args.comment)

    if not per_file:
        print("No valid data found in the provided files.")
        return

    print("\n" + "="*45)
    print(f"{'FILE NAME':<30} | {'INSTANCES'}")
    print("-" * 45)
    for file, count in per_file.items():
        display_name = (file[:47] + '..') if len(file) > 49 else file
        print(f"{display_name:<50} | {count}")

    print("-" * 60)
    print(f"TOTAL INSTANCES (CODE ONLY): {total}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
