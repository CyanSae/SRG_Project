import os
import glob

def count_lines_of_code(directory):
    total_lines = 0

    for filepath in glob.glob(os.path.join(directory, '**', '*.py'), recursive=True):
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            total_lines += len(lines)
            print(f"{filepath}: {len(lines)} lines")
    
    print(f"Total lines of code in directory '{directory}': {total_lines}")

directory_path = '/home/sandra/projects/GCN'
count_lines_of_code(directory_path)