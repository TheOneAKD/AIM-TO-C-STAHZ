import pandas as pd
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView
import os
from astropy.io import fits
import matplotlib.pyplot as plt

# File paths
useful_names_file = "data/useful_names.txt"
output_csv_file = "data/star_dataset.csv"
image_dir = r"C:\Users\Jyoti\OneDrive\Desktop\Coding\SciRe 2024-25 AIM-TO-C-STAHZ\images"
os.makedirs(image_dir, exist_ok=True)  # Ensure image directory exists

# Initialize Simbad fields for required stats
Simbad.add_votable_fields(
    'flux(V)', 'otype', 'sptype', 'distance', 'pm', 'rvz_radvel', 'fe_h', 'diameter'
)

# Check if the useful names file exists
if not os.path.exists(useful_names_file):
    print(f"File {useful_names_file} not found. Exiting...")
    exit()

# Read useful star names from the file
def read_names_from_file(file_path):
    """Read star names from the file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# ====== Functions for Derived Calculations ======
def compute_mass(spectral_type):
    """Estimate mass based on spectral type (approximation)."""
    mapping = {'O': 40, 'B': 8, 'A': 3, 'F': 1.7, 'G': 1.1, 'K': 0.8, 'M': 0.3}
    return mapping.get(spectral_type[0], None)

def compute_age(spectral_type):
    """Estimate age based on spectral type and lifetime models."""
    mapping = {'O': 1e6, 'B': 1e7, 'A': 5e8, 'F': 2e9, 'G': 10e9, 'K': 15e9, 'M': 20e9}
    return mapping.get(spectral_type[0], None)

# ====== Image Fetching ======
def fetch_star_image(star_name):
    """Fetch an image for the star using SkyView."""
    try:
        images = SkyView.get_images(position=star_name, survey='DSS', pixels='300,300')
        if images:
            fits_data = images[0][0]
            img_name = os.path.join(image_dir, f"{star_name.replace(' ', '_')}.jpg")

            # Save the FITS data as an image
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

# ====== Main Star Info Fetch ======
def fetch_star_info(star_name):
    """Query Simbad for star information and fetch its image."""
    try:
        result = Simbad.query_object(star_name)
        if result:
            spectral_type = result['SP_TYPE'][0] if 'SP_TYPE' in result.colnames else None
            luminosity = result['FLUX_V'][0] if 'FLUX_V' in result.colnames else None
            apparent_magnitude = result['FLUX_V'][0] if 'FLUX_V' in result.colnames else None
            distance = result['Distance_distance'][0] if 'Distance_distance' in result.colnames else None
            diameter = result['Diameter_diameter'][0] if 'Diameter_diameter' in result.colnames else None
            proper_motion_ra = result['PMRA'][0] if 'PMRA' in result.colnames else None
            proper_motion_dec = result['PMDEC'][0] if 'PMDEC' in result.colnames else None
            radial_velocity = result['RVZ_RADVEL'][0] if 'RVZ_RADVEL' in result.colnames else None
            metallicity = result['Fe_H_Fe_H'][0] if 'Fe_H_Fe_H' in result.colnames else None
            temperature = {'O': 30000, 'B': 20000, 'A': 10000, 'F': 7500, 'G': 5500, 'K': 4500, 'M': 3000}.get(
                spectral_type[0], None) if spectral_type else None

            if spectral_type and luminosity and distance and apparent_magnitude and temperature:
                mass = compute_mass(spectral_type)
                age = compute_age(spectral_type)

                # Fetch the image for the star
                image_path = fetch_star_image(star_name)

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
            print(f"No data found for star: {star_name}")
            return None
    except Exception as e:
        print(f"Error fetching data for {star_name}: {e}")
        return None

# Main processing function
def generate_star_dataset(input_file, output_file):
    """Generate a dataset CSV file for useful stars."""
    star_names = read_names_from_file(input_file)
    data = []

    for idx, star_name in enumerate(star_names, start=1):
        print(f"Processing {idx}/{len(star_names)}: {star_name}")
        star_info = fetch_star_info(star_name)
        if star_info:
            data.append(star_info)

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
    else:
        print("No valid data collected. Dataset not created.")

# Run the program
if __name__ == "__main__":
    generate_star_dataset(useful_names_file, output_csv_file)
