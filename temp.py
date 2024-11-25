import os

# Path to the images folder
image_dir = r"C:\Users\Jyoti\OneDrive\Desktop\Coding\SciRe 2024-25 AIM-TO-C-STAHZ\images"

# Output file to save the names
output_file = "data/useful_names.txt"

def get_star_names_from_images(image_dir, output_file):
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

# Run the script
get_star_names_from_images(image_dir, output_file)


from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_star_coordinates(star_name):
    """
    Get the tuple (RA, Dec) of a star's coordinates.
    
    Parameters:
        star_name (str): Name of the star.
    
    Returns:
        tuple: (RA in degrees, Dec in degrees) or None if the query fails.
    """
    try:
        # Query Simbad for star information
        result = Simbad.query_object(star_name)
        if result:
            ra = result['RA'][0]  # Right Ascension (as string, e.g., "10 45 03.591")
            dec = result['DEC'][0]  # Declination (as string, e.g., "+59 41 04.26")

            # Convert RA/Dec to degrees
            coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            return (coords.ra.deg, coords.dec.deg)
        else:
            print(f"No coordinates found for {star_name}")
            return None
    except Exception as e:
        print(f"Error fetching coordinates for {star_name}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    star_name = "HD_189022"
    coordinates = get_star_coordinates(star_name)
    if coordinates:
        print(f"Coordinates for {star_name}: {coordinates}")
