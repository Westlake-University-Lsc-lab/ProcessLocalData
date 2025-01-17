import os
import runinfo

def main():
    parser = argparse.ArgumentParser(description='Modify file name')
    parser.add_argument('--file-tag-str', type=str, required=True, help='File tag string')
    parser.add_argument('--modify-str', type=str, required=True, help='Modify string')
    parser.add_argument('--target-str', type=str, required=True, help='Target string')
    args = parser.parse_args()
    
    runinfo.modify_file_names(args.target_str, args.modify_str, args.file_tag_str)

if __name__ == '__main__':
    main()