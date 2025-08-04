import argparse
import os

import json
import mimetypes
from pillow_heif import register_heif_opener

import exif
import geo
import geotagger_agent


def find_image_files(directory_path: str) -> list[str]:
    """
    Scans a given directory and its subdirectories for image files.

    Args:
        directory_path (str): The path to the directory to scan.

    Returns:
        list[str]: A list of absolute paths to image files found.
    """
    image_info_list = []

    # Ensure the directory exists and is actually a directory
    if not os.path.isdir(directory_path):
        print(
            f"Error: '{directory_path}' is not a valid directory or does not exist.")
        return []

    register_heif_opener()

    # Walk through the directory tree
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Guess the MIME type based on the file extension
            # mimetypes.guess_type returns a tuple (type, encoding)
            mime_type, _ = mimetypes.guess_type(file_path)

            # Check if the guessed MIME type indicates an image
            if mime_type and mime_type.startswith('image/'):
                exif_data = exif.get_exif_data(file_path)
                lat, lng = exif.get_lat_lng(exif_data)
                image_info_list.append({
                    'filepath': file_path,
                    'datetime': exif.get_datetime(exif_data),
                    'location': geo.get_place_name(lat, lng),
                })
        # Do not go into subfolders
        break

    return image_info_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="The path to the directory of images."
    )

    args = parser.parse_args()

    target_directory = os.path.abspath(args.filepath)

    print(f"Scanning '{target_directory}' for image files...")
    images = find_image_files(target_directory)
    images.sort(key=lambda x: x['datetime'])
    for image in images:
        image['datetime'] = image['datetime'].strftime("%Y-%m-%d %H:%M:%S%z")

    print('Found the following images:')
    print(images)

    agent_executor = geotagger_agent.get_geotagger_agent()

    # The agent executor takes a dictionary input
    response = agent_executor.invoke({"input": json.dumps(images)})

    # # The final answer is in the 'output' key
    print("AI:", response['output'])
    print(type(response['output']))
    ai_results = json.loads(response['output'])

    BOLD = '\033[1m'
    # ANSI escape code to reset formatting (important!)
    END = '\033[0m'
    changed_locations = {}
    for image in images:
        changed_location = None
        if image['location'] is None:
            # See if we found a result.
            for result in ai_results:
                if result['filepath'] == image['filepath'] and result['location'] is not None:
                    changed_location = result['location']
                    changed_locations[result['filepath']] = changed_location
                    break
        if changed_location:
            image_output = f"{image['filepath']} {image['datetime']} {BOLD}updated location {changed_location}{END}"
        else:
            image_output = f"{image['filepath']} {image['datetime']} {image['location']}"
        print(image_output)

    user_input_commit = input(
        'These are the proposed location assignments for the images. Do you want to commit them? Y/n').strip()
    print(user_input_commit)

    if user_input_commit != 'Y':
        print('Not commiting changes. Terminating program.')
        exit()

    # Commit changes to EXIF.
    for filepath, changed_location in changed_locations.items():
        print(filepath, changed_location)
        lat, lng = geo.get_lat_lng(changed_location)
        exif.set_coordinates_exif(filepath, lat, lng)
