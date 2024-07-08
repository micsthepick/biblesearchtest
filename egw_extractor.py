import zipfile
import glob
import json
import os
from tqdm.auto import tqdm

def peek_and_extract_info_from_egwbooks():
    # Initialize an empty list to store the contents of info.json files
    info_contents = []

    # Use glob to find all .egwbook files (treated as ZIP files)
    egwbook_files = glob.glob("egwbooks/*.egwbook")

    # Iterate over each .egwbook file
    for egwbook_file in tqdm(egwbook_files):
        # Open the ZIP file
        with zipfile.ZipFile(egwbook_file, 'r') as zip_ref:
            # List all files in the ZIP
            file_list = zip_ref.namelist()

            # Check if info.json exists in the ZIP
            if 'info.json' in file_list:
                # Extract the contents of info.json
                with zip_ref.open('info.json') as info_file:
                    # Read the contents and parse as JSON
                    info_data = json.load(info_file)
                    # Append the parsed JSON data to the info_contents list
                    info_contents.append(info_data)

    # Write the array of info.json contents to books.json
    with open('egwbooks/books.json', 'w') as books_json_file:
        json.dump(info_contents, books_json_file, indent=4)

if __name__ == "__main__":
    peek_and_extract_info_from_egwbooks()