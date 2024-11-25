import pandas as pd
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== Utility functions for file operations ======
def read_names_from_file(file_path):
    """Read names from a file into a set."""
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def get_file_length(file_path):
    """Return the number of lines in a file."""
    with open(file_path, 'r') as f:
        return sum(1 for line in f if line.strip())  # Count non-empty lines

def append_name_to_file(file_path, name):
    """Append a name to a file if not already present."""
    with open(file_path, 'r+') as f:
        existing_names = set(line.strip() for line in f if line.strip())  # Read and strip existing names
        if name not in existing_names:
            f.write(name + '\n')
        else:
            print(f"{name} already exists in {file_path}.")

def load_visited_coordinates(visited_coords_file='data/visited_coords.txt'):
    """Load visited coordinates from file into a set."""
    with open(visited_coords_file, 'r') as f:
        return set(tuple(map(float, line.strip().split(','))) for line in f if line.strip())

def save_visited_coordinate(ra, dec, visited_coords_file='data/visited_coords.txt'):
    """Save a visited coordinate to the file if not already present."""
    coord_str = f"{ra},{dec}"
    with open(visited_coords_file, 'r+') as f:
        existing_coords = set(line.strip() for line in f if line.strip())
        if coord_str not in existing_coords:
            f.write(coord_str + '\n')
        else:
            print(f"Coordinate {coord_str} already exists in {visited_coords_file}.")

def generate_star_dataset(input_file='useful_names.txt', output_file='data/star_dataset.csv'):
    """Generate a dataset CSV file for useful stars."""
    star_names = read_names_from_file(input_file)
    data = []

    for idx, star_name in enumerate(star_names, start=1):
        print(f"Processing {idx}/{len(star_names)}: {star_name}")
        star_info = query_star_info(star_name)
        if star_info:
            data.append(star_info)

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
    else:
        print("No valid data collected. Dataset not created.")

def get_star_names_from_images(image_dir='data/images', output_file='data/star_dataset.csv'):
    """Extract star names from image file names and save to a text file."""
    # List to hold the star names
    star_names = []

    # Check if the directory exists
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist.")
        return

    # Loop through all files in the directory
    for file_name in os.listdir(image_dir):
        # Skip hidden/system files
        if file_name.startswith('.'):
            continue

        # Get the star name by removing the file extension
        star_name, _ = os.path.splitext(file_name)
        star_names.append(star_name)

    # Save the star names to the output file
    with open(output_file, 'w') as f:
        for name in star_names:
            f.write(name + '\n')

    print(f"Star names extracted and saved to {output_file}")

def check_file_for_duplicates(file_path):
    """Check if a file contains duplicate entries."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines and strip whitespace
        unique_lines = set(lines)  # Convert to a set to remove duplicates

        if len(lines) != len(unique_lines):
            print(f"Duplicates found in {file_path}:")
            duplicates = [line for line in lines if lines.count(line) > 1]
            duplicates = list(set(duplicates))  # Get unique duplicates
            for dup in duplicates:
                print(f"  Duplicate: {dup}")
        else:
            print(f"No duplicates found in {file_path}")

def remove_duplicates_from_file(file_path):
    """Remove duplicate entries from a file and save the unique entries back to the file."""
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]  # Read and clean lines
            unique_lines = sorted(set(lines))  # Get unique lines and sort them for consistency

        # Overwrite the file with unique lines
        with open(file_path, 'w') as f:
            for line in unique_lines:
                f.write(line + '\n')

        print(f"Removed duplicates from {file_path}. {len(lines) - len(unique_lines)} duplicates removed.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def find_highest_ra_dec(file_path='data/visited_coords.txt'):
    """
    Find the pair with the highest RA and Dec from a visited coordinates file.

    Args:
        file_path (str): Path to the visited coordinates file.

    Returns:
        tuple: The coordinate pair with the highest RA and Dec, in the format (RA, Dec).
    """
    highest_ra = float('-inf')
    highest_dec = float('-inf')
    highest_pair = None

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    ra, dec = map(float, line.split(','))
                    if dec > highest_dec or (dec == highest_dec and ra > highest_ra):
                        highest_ra = ra
                        highest_dec = dec
                        highest_pair = (ra, dec)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    return highest_pair

# ====== Functions for Derived Calculations ======
def compute_mass(spectral_type):
    """Estimate mass based on spectral type (approximation)."""
    mapping = {'O': 40, 'B': 8, 'A': 3, 'F': 1.7, 'G': 1.1, 'K': 0.8, 'M': 0.3}
    return mapping.get(spectral_type[0], None)

def compute_age(spectral_type):
    """Estimate age based on spectral type and lifetime models."""
    mapping = {'O': 1e6, 'B': 1e7, 'A': 5e8, 'F': 2e9, 'G': 10e9, 'K': 15e9, 'M': 20e9}
    return mapping.get(spectral_type[0], None)

# ====== Intermediate Steps for Star Collection ======
def fetch_star_image(star_name, image_dir='data/images'):
    """Fetch and save an image of the star."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use a non-GUI backend
        # Query SkyView for images centered on the star
        images = SkyView.get_images(position=star_name, survey='DSS', pixels='300,300')
        if images:
            fits_data = images[0][0]
            img_name = os.path.join(image_dir, f"{star_name.replace(' ', '_')}.jpg")
            
            # Open the FITS file and plot it
            data = fits_data.data
            plt.figure()
            plt.imshow(data, cmap='gray', origin='lower')
            plt.axis('off')
            plt.savefig(img_name, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            print(f"Image saved as {img_name}")
            return img_name
        else:
            print(f"No image found for {star_name}")
            return None
    except Exception as e:
        print(f"Failed to fetch image for {star_name}: {e}")
        return None

    
def query_star_info(star_name, names_missing_info, names_missing_img, useful_names, useful_names_file='data/useful_names.txt', missing_info_file='data/x_info_names.txt', missing_img_file='data/x_img_names.txt'):
    """
    Query detailed information about a star from the SIMBAD database.

    This function retrieves specific attributes of a star from the SIMBAD database,
    including essential physical properties and additional metadata. It also fetches
    an image of the star and computes derived attributes like mass and age.

    Args:
        star_name (str): The name of the star to query.
        names_missing_info (int): Counter for stars missing required information.
        names_missing_img (int): Counter for stars missing associated images.
        useful_names (int): Counter for stars successfully processed.
        useful_names_file (str, optional): File path to store names of useful stars. Defaults to 'data/useful_names.txt'.
        missing_info_file (str, optional): File path to log stars with missing information. Defaults to 'data/x_info_names.txt'.
        missing_img_file (str, optional): File path to log stars with missing images. Defaults to 'data/x_img_names.txt'.

    Returns:
        dict: A dictionary containing the star's attributes, including:
            - "Star Name": The name of the star.
            - "Spectral Type": The spectral classification (e.g., OBAFGKM scale).
            - "Luminosity": The star's luminosity in flux(V).
            - "Apparent Magnitude": The apparent brightness from Earth.
            - "Distance": Distance from Earth (e.g., in parsecs).
            - "Diameter": Diameter of the star (if available).
            - "Proper Motion RA": Proper motion in RA (if available).
            - "Proper Motion DEC": Proper motion in DEC (if available).
            - "Radial Velocity": The star's radial velocity (if available).
            - "Metallicity": The metallicity of the star (if available).
            - "Temperature": Estimated temperature based on the spectral type.
            - "Mass": Estimated mass based on the spectral type.
            - "Age": Estimated age based on the spectral type.

        None: If the star lacks critical information or an error occurs.

    Raises:
        Exception: Catches and logs errors during SIMBAD queries or image fetching.

    Notes:
        - Logs stars with missing information or images in respective files.
        - Updates counters for successfully processed stars and missing data.
        - Uses the `fetch_star_image` function to retrieve and save the star's image.
        - Derived values like mass and age are approximations based on spectral type.

    Example:
        >>> query_star_info("Sirius", 0, 0, 0)
        {
            "Star Name": "Sirius",
            "Spectral Type": "A1V",
            "Luminosity": 1.42,
            "Apparent Magnitude": -1.46,
            "Distance": 2.64,
            "Diameter": 1.9,
            "Proper Motion RA": -546.01,
            "Proper Motion DEC": -1223.08,
            "Radial Velocity": -5.5,
            "Metallicity": -0.1,
            "Temperature": 9900,
            "Mass": 2.1,
            "Age": 0.5e9,
        }
    """

    try:
        result = Simbad.query_object(star_name)
        if result:
            # ====== MUST HAVE ====== #
            spectral_type = result['SP_TYPE'][0] if 'SP_TYPE' in result.colnames else None
            luminosity = result['FLUX_V'][0] if 'FLUX_V' in result.colnames else None
            apparent_magnitude = result['FLUX_V'][0] if 'FLUX_V' in result.colnames else None
            distance = result['Distance_distance'][0] if 'Distance_distance' in result.colnames else None
            diameter = result['Diameter_diameter'][0] if 'Diameter_diameter' in result.colnames else None
            # ====== NOT AS IMPORTANT ====== #
            proper_motion_ra = result['PMRA'][0] if 'PMRA' in result.colnames else None
            proper_motion_dec = result['PMDEC'][0] if 'PMDEC' in result.colnames else None
            radial_velocity = result['RVZ_RADVEL'][0] if 'RVZ_RADVEL' in result.colnames else None
            metallicity = result['Fe_H_Fe_H'][0] if 'Fe_H_Fe_H' in result.colnames else None
            temperature = {'O': 30000, 'B': 20000, 'A': 10000, 'F': 7500, 'G': 5500, 'K': 4500, 'M': 3000}.get(
                spectral_type[0], None) if spectral_type else None

            if spectral_type and luminosity and apparent_magnitude and distance and diameter:
                # Fetch the image for the star
                image_path = fetch_star_image(star_name)
                if image_path:  # Only accept stars with images
                    mass = compute_mass(spectral_type)
                    age = compute_age(spectral_type)

                    useful_names += 1
                    append_name_to_file(useful_names_file, star_name)
                    return {
                        "Star Name": star_name,
                        "Spectral Type": spectral_type,
                        "Luminosity": luminosity,
                        "Apparent Magnitude": apparent_magnitude,
                        "Distance": distance,
                        "Diameter": diameter,
                        "Proper Motion RA": proper_motion_ra,
                        "Proper Motion DEC": proper_motion_dec,
                        "Radial Velocity": radial_velocity,
                        "Metallicity": metallicity,
                        "Temperature": temperature,
                        "Mass": mass,
                        "Age": age,
                    }
                else:
                    names_missing_img += 1
                    append_name_to_file(missing_img_file, star_name)
                    return None
            else:
                names_missing_info += 1
                append_name_to_file(missing_info_file, star_name)
                return None
        else:
            names_missing_info += 1
            append_name_to_file(missing_info_file, star_name)
            return None
    except Exception as e:
        print(f"Error querying star {star_name}: {e}")
        return None

def get_stars_in_region(ra, dec, visited_coordinates, radius):
    """
    Retrieve stars in a specified region using SIMBAD's cone search.

    This function queries the SIMBAD database for stars within a specified region
    centered on given Right Ascension (RA) and Declination (Dec) coordinates. It
    checks if the coordinates have already been visited, and if not, performs the 
    query. The function only returns stars classified as such in the database.

    Args:
        ra (float): The Right Ascension (RA) in degrees of the region's center.
        dec (float): The Declination (Dec) in degrees of the region's center.
        visited_coordinates (set): A set of previously visited coordinate tuples 
                                   to avoid duplicate queries.
        radius (float): The radius of the search region in degrees.

    Returns:
        list: A list of star names within the specified region. If no stars are 
              found or an error occurs, an empty list is returned.

    Raises:
        Exception: If an error occurs during the SIMBAD query, it is caught, and
                   an error message is printed.

    Notes:
        - Updates the `visited_coordinates` set to include the queried region.
        - Uses the `save_visited_coordinate` function to persist visited coordinates.

    Example:
        >>> get_stars_in_region(ra=10.0, dec=-5.0, visited_coordinates=set(), radius=1.0)
        ["Sirius", "Betelgeuse"]
    """

    if (ra, dec) in visited_coordinates:
        print(f"Skipping already visited coordinates: RA={ra}, Dec={dec}")
        return []

    try:
        coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
        result = Simbad.query_region(coords, radius=radius * u.degree)
        save_visited_coordinate(ra, dec)
        coord_str = f"({ra},{dec})"
        visited_coordinates.add(coord_str)
        if result and len(result) > 0:
            stars = [name.strip() for name, otype in zip(result['MAIN_ID'], result['OTYPE']) if otype == 'Star']
            return stars
        else:
            return []
    except Exception as e:
        print(f"Error querying stars at RA={ra}, Dec={dec}: {e}")
        return []

def process_star_batch(star_batch, names_missing_info, names_missing_img, useful_names, processed_names, processed_missing_img_names, processed_missing_info_names, processed_useful_names, required_rows, all_names, ra, dec, all_names_file='data/all_names.txt'):
    """
    Process a batch of star names to collect detailed information and update the dataset.

    This function iterates through a batch of star names, checks if they have been processed before, and queries
    information about unprocessed stars. It updates the counters and logs as necessary.

    Args:
        star_batch (list): List of star names to process.
        processed_names (set): Set of all previously processed star names.
        processed_missing_img_names (set): Set of stars missing associated images.
        processed_missing_info_names (set): Set of stars missing required information.
        processed_useful_names (set): Set of stars successfully added to the dataset.
        useful_names (int): Counter for stars successfully processed and added to the dataset.
        required_rows (int): Total number of useful stars required for the dataset.
        all_names (int): Counter for all stars processed (including unuseful ones).
        ra (float): Current right ascension (RA) coordinate being processed.
        dec (float): Current declination (Dec) coordinate being processed.
        all_names_file (str, optional): File path to store the names of all processed stars. Defaults to 'data/all_names.txt'.

    Returns:
        list: A list of dictionaries containing detailed information about stars successfully added to the dataset.

    Notes:
        - The function skips stars that are already in processed sets or marked as missing information or images.
        - Logs star names to `all_names_file` as they are processed.
        - Queries detailed star information using the `query_star_info` function.
        - Adds successfully processed stars to the dataset and returns their details.

    Example:
        >>> process_star_batch(
        ...     star_batch=["Sirius", "Betelgeuse"],
        ...     processed_names=set(),
        ...     processed_missing_img_names=set(),
        ...     processed_missing_info_names=set(),
        ...     processed_useful_names=set(),
        ...     useful_names=0,
        ...     required_rows=100,
        ...     all_names=0,
        ...     ra=10.0,
        ...     dec=-5.0
        ... )
        [
            {
                "Star Name": "Sirius",
                "Spectral Type": "A1V",
                "Luminosity": 1.42,
                ...
            },
            {
                "Star Name": "Betelgeuse",
                "Spectral Type": "M1Ia",
                "Luminosity": 140000,
                ...
            }
        ]
    """
    data = []

    for star_name in star_batch:
        if star_name in processed_names or star_name in processed_missing_img_names or star_name in processed_missing_info_names:
            print(f"Skipping already processed star: {star_name}")
            continue
        if star_name in processed_useful_names:
            print(f"Useful star {star_name} found, but already in dataset; skipping.")
            continue

        all_names += 1
        append_name_to_file(all_names_file, star_name)
        print(f"{useful_names}/{required_rows} | Processing Star #{all_names}/{get_file_length(all_names_file)} at ({ra},{dec}): {star_name}")
        
        star_info = query_star_info(star_name, names_missing_info, names_missing_img, useful_names)
        if star_info:
            data.append(star_info)
    
    return data
