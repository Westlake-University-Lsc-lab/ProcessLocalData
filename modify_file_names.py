import os
import argparse
import sys

def modify_file_names(target_date_str, modify_string, new_date_str):
    search_path = "/mnt/data/PMT/R8520_406/"
    if not os.access(search_path, os.F_OK):
        raise ValueError("Path not found or inaccessible: " + search_path)
    for root, dirs, files in os.walk(search_path):
        for filename in files:
            if target_date_str in filename:
                new_filename = filename.replace(modify_string, new_date_str)
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)
                print("--------------------")                
                print("New filename:", new_file_path)
    return 0

def main():
    try:
        parser = argparse.ArgumentParser(description="Modify file names in directory /mnt/data/PMT/R8520_406/")
        parser.add_argument("--target_date_str", type=str, help="Target date string in the file names, usually in the format of 20250121")
        parser.add_argument("--modify_str", type=str, help="String to be modified in the file names, usually run_tag_str")
        parser.add_argument("--new_str", type=str, help="New date string to be used in the file names, usually run_tag_str")
        args = parser.parse_args()
        if len(vars(args)) != 3:
            raise ValueError("Invalid arguments")
            print("Usagee: python modify_file_names.py --target_date_str 20250121 --modify_str  low_resistor --new_str resistor_12p5M ")
            sys.exit(1)
        modify_file_names(args.target_date_str, args.modify_str, args.new_str)
    except Exception as e:
        print(e)
        print("Usagee: python modify_file_names.py --target_date_str 20250121 --modify_str  low_resistor --new_str resistor_12p5M ")

if __name__ == "__main__":
    main()