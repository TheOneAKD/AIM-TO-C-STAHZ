import pandas as pd
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    read_names_from_file,
    append_name_to_file,
    load_visited_coordinates,
    save_visited_coordinate,
    get_file_length,    
    get_stars_in_region,
    query_star_info,
    process_star_batch,
    find_highest_ra_dec,
)

# Define the directory to save images
image_dir = r"C:\Users\Jyoti\OneDrive\Desktop\Coding\SciRe 2024-25 AIM-TO-C-STAHZ\images"
os.makedirs(image_dir, exist_ok=True)  # Create directory if it doesn't exist

# Define file paths for saving names and visited coordinates
global all_names_file, useful_names_file, missing_img_file, missing_info_file, visited_coords_file, dataset_file
all_names_file = "data/all_names.txt"
useful_names_file = "data/useful_names.txt"
missing_img_file = "data/x_img_names.txt"
missing_info_file = "data/x_info_names.txt"
visited_coords_file = "data/visited_coords.txt"
dataset_file = "data/star_dataset.csv"

# Ensure files exist
for file in [useful_names_file, all_names_file, missing_img_file, missing_info_file, visited_coords_file]:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            pass  # Create empty file

# Define a custom Simbad query to get specific details about stars
Simbad.add_votable_fields(
    'flux(V)', 'otype', 'sptype', 'distance', 'pm', 'rvz_radvel', 'fe_h', 'diameter'
)

# Dynamic user inputs
global required_rows
while True:

    # Get only useful stars that have not been analyzed yet without adding extra datapoints
    with open(useful_names_file, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)

    choice = input("Add x to dataset (a), have x in total (b): ")
    if choice == "a":
        required_rows = int(input("Enter the number of stars to ADD to the dataset: "))
        break
    elif choice == "b":
        required_rows = int(input("Enter the number of stars to be IN the dataset: "))
        if required_rows <= line_count:
            continue
        required_rows -= line_count
        break

global radius, offset_step, all_names, useful_names, names_missing_img, names_missing_info

radius = 1.0
offset_step = 2.0  # Offset step as 2x the radius

visited_coordinates = load_visited_coordinates()

# Other important values
initial_ra, initial_dec = find_highest_ra_dec()  # Starting coordinates
ra, dec = find_highest_ra_dec()
max_threads = 24  # Number of parallel threads

# COUNTERS
all_names = 0
useful_names = 0
names_missing_img = 0
names_missing_info = 0

# ====== MAIN LOGIC ====== #
def compile_dataset(required_count):
    """Compile the dataset of stars with complete information."""
    global useful_names, all_names, processed_names, processed_useful_names, processed_missing_img_names, processed_missing_info_names, visited_coordinates, ra, dec, dataset_file
    data = []
    current_ra, current_dec = initial_ra, initial_dec

    with ThreadPoolExecutor(max_threads) as executor:
        future_to_ra_dec = {}

        while len(data) < required_count:
            # Refresh processed sets and visited coordinates
            processed_useful_names = read_names_from_file(useful_names_file)
            processed_names = read_names_from_file(all_names_file)
            processed_missing_img_names = read_names_from_file(missing_img_file)
            processed_missing_info_names = read_names_from_file(missing_info_file)
            visited_coordinates = load_visited_coordinates()

            for _ in range(max_threads):
                future = executor.submit(get_stars_in_region, current_ra, current_dec, visited_coordinates, radius)
                future_to_ra_dec[future] = (current_ra, current_dec)
                ra += offset_step
                current_ra += offset_step
                if current_ra >= 360.0:
                    current_ra = 0.0
                    ra = 0.0
                    dec += offset_step
                    current_dec += offset_step
                    print(f"\nReached max RA, resetting and incrementing DA. Currently have {useful_names}/{required_rows} useful stars out of {all_names} stars processed.\n")
                    if current_dec > 90.0:
                        print(f"Reached the end of the declination range. Program terminating with {useful_names}/{required_rows} useful stars out of {all_names} stars processed.")
                        break

            for future in as_completed(future_to_ra_dec):
                ra, dec = future_to_ra_dec[future]
                coord_str = f"{ra},{dec}"
                if coord_str in visited_coordinates:
                    continue
                try:
                    star_batch = future.result()
                    if not star_batch:
                        print(f"No stars found at RA={ra}, Dec={dec}.")
                        save_visited_coordinate(ra, dec)
                        continue

                    batch_data = process_star_batch(star_batch, names_missing_info, names_missing_img, useful_names, processed_names, processed_missing_img_names, processed_missing_info_names, processed_useful_names, required_rows, all_names, ra, dec)
                    data.extend(batch_data)
                    save_visited_coordinate(ra, dec)
                    if len(data) >= required_count:
                        break
                except Exception as e:
                    print(f"Error processing region RA={ra}, Dec={dec}: {e}")

            if current_dec > 90.0:
                break

    # Save the dataset
    output_file = dataset_file
    if os.path.exists(output_file):
        os.remove(output_file)

    pd.DataFrame(data).to_csv(output_file, index=False)
    print(f"Dataset compiled with {len(data)} stars and saved as {output_file}")

    print(f"\nSummary:\nTotal Names Processed: {all_names}")
    print(f"    Names Missing Required Information: {names_missing_info}")
    print(f"    Names Missing Image: {names_missing_img}")
    print(f"    Names In Dataset: {useful_names}")

if __name__ == "__main__":
    compile_dataset(required_rows)