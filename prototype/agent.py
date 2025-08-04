import argparse
import base64
from datetime import datetime
import mimetypes
import os
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.memory import ConversationBufferMemory
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any
import json
import googlemaps

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


def analyze_locations_by_time(metadata):
    print('analyzing locations by time')
    print(metadata)
    metadata = json.loads(metadata)

    agent_prompt = f"""You are trying to figure out the location of where each photo in a list was taken. You know the timestamp of when each photo was taken. But you do not know the location. You do know the location and timestamp of the most recent photo taken before the list of photos with unknown locations and the most recent photo taken after the list of photos with unkown location. Your job is to use the timestamps to assess whether any if not all of the photos might be taken at the same location. For example, if a photo was taken at 9pm on 2025-01-01 with known location at Times Square, it is quite likely that a photo taken at 9:05pm on 2025-01-01 was also taken at Time Square.

        The list of photos is: {metadata}

        Note that each photo contains fields representing the filepath, location, and datetime. Photos with a null or empty location are the ones with unknown locations - these are the photos whose location you are trying to figure out.

        Give your answer in a list. The list should contain a prediction of the location for each unknown photo. The prediction should either give a location or the term "unknown" if you cannot determine the location from the provided information. The element in the input list of intemediary photo_times should match the corresponding index in the response list. If you are unable to predict whether the photo is at the specific location, you can give a prediction on whether the photo is at the same broader location. For example, if you cannot know for sure the photo was taken in Times Square, but you can assess the photo was taken in New York City, then you can give New York city as a prediction. However, you should still predict unknown if you cannot accurately predict the location at the level or more granular level than the city.

    """

    print(f"---CALLING GEMINI with query: '{agent_prompt}'---")

    response = llm.invoke(agent_prompt)
    return response.content


def analyze_locations_by_photo(image_paths):
    print('analyzing_locations_by_photo')
    print(image_paths)
    image_paths = json.loads(image_paths)
    image_datas = []

    for image_path in image_paths:
        mime_type, _ = mimetypes.guess_type(image_path)
        print('mime type', mime_type, image_path)
        if mime_type is None or not mime_type.startswith('image/'):
            raise RuntimeError(
                f"Error: Could not determine image MIME type for {image_path} or it's not an image.")

        # Read the image file in binary mode and encode to base64
        with open(image_path, "rb") as image_file:
            encoded_image_data = base64.b64encode(
                image_file.read()).decode("utf-8")
            image_datas.append(
                {'mime_type': mime_type, 'image': encoded_image_data})

    llm_vision = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.0)

    # content=[
    #     {"type": "text", "text": "Given the following list of images, can you predict whether or not they were taken at the same location? Respond with a list containing the same number of elements as the input list. Each element should indicate the index number of another element that you believe were taken at the same location. You should use an index of -1 if the image was not taken at the same location as any other photo in the list, or if there is insufficient information. Afterwards, state your reasoning for each index."},
    # ]
    # content=[
    #     {"type": "text", "text": "Given the following list of images, can you predict whether or not they were all taken at the same location? You should respond in one of the following ways: 1) same precise location - respond this way if you are confident the images were taken at the exact location, for example, if all the images were taken at the Statue of Liberty; 2) same general location - respond this way if you are confident the images are taken at the same broader location, for example, if all the images were taken in New York City; 3) same type of location - respond this way if you are confident the images are taken at the same type of location but you cannot be certain they were taken at the same place, for example, if all the images were taken in a city as opposed to in nature; or 4) different locations/inconclusive. Give the response in the JSON format. There should be one key called 'response' where you give the above answer, and there should be a second key called 'reason' where you explain your answer."},
    # ]
    content = [
        {"type": "text", "text": "Given the following list of images, can you predict where each image was taken? You should give as precise of a location as you are confident is accurate. This also means you should give a broader location if you are not confident of a more precise location. An example of the increasing broadness of location granularities is: 1) exact location, e.g. Statue of Liberty; 2) broader location, e.g. New York City; 3) type of location, e.g. image was taken in a big city; and 4) uncertain where this image was taken."},
    ]
    for image_data in image_datas:
        content.append({
            "type": "image_url",  # Use image_url type for inline base64 in LangChain
            "image_url": {
                "url": f"data:{image_data['mime_type']};base64,{image_data['image']}"
            }
        })

    message = HumanMessage(
        content=content
    )
    response = llm_vision.invoke([message])
    return response.content


tools = [
    Tool(
        name="AnalyzeLocationsByTime",
        func=analyze_locations_by_time,
        description="Useful for when you want to determine which photos are likely to be taken at the same location as other photos, purely based on the time they were taken at. You should input a list of images. Each image should be a dictionary with the filepath, location, and datetime. Images where we don't know the location will have a null location value.",
    ),
    Tool(
        name="AnalyzeLocationsByPhoto",
        func=analyze_same_location_by_picture,
        description="Useful for when you want to determine where a photo was taken based on the image itself, and/or if 2 or more photos were taken at the same location. You should input the image_paths of the photos.",
    )
]


template = """
You are trying to figure out the location of where each photo in a list was taken. Some photos in the list already have a known location but some photos do not have a known location. All photos in the list have a known timestamp. You must try to determine the location of the photos with unknown locations by using the tools below.

You have the following tools you can use: {tools}

You must use all the tools once. You should use these tools in conjuction with each other to figure out the locations of the photos.

When using AnalyzeLocationsByPhoto, do not accept a location that is not in the list of photos with known locations. You should use the description of the location from AnalyzeLocationsByPhoto combined with photos taken at similar timestamps (as per AnalyzeLocationsByTime) to figure out either a precise or a broader location for the images. If you are not confident of a precise or broader location, it's OK to leave the photo's location as unknown.

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

You should compare the answers from each tool you use and analyze the results in conjucation with each other. Once you are confident in the location of the photos, or if you are confident that the location cannot be determined, you should return the response to the user.

When you have a response, or if you do not need to use a tool, you must use this format:

Final Answer: [your answer]

You should return the answer in JSON format. Do not add any special formatting such as ```json. Make sure the response can be parsed by Python's json.loads() function. The response should be a list of objects, corresponding to the images that you have found locations for. Each object should contain the following fields:
filepath: the image file path
timestamp: the timestamp of the image
location: the string location of the image

You should return all images, even if the image already had a location assigned or if you weren't successful in finding a location for that image.

When passing data into subagents, make sure to use the JSON format if you are passing multiple variables and/or parameters.

The inputs are below:

{input}

{agent_scratchpad}

"""

# Define the prompt template
prompting = PromptTemplate(template=template)

# Add memory to the agent to remember conversation history
memory = ConversationBufferMemory(memory_key="chat_history")

# Create the agent itself. This binds the LLM, prompt, and tools.
# The agent is the "supervisor" logic.
agent = create_react_agent(llm, tools, prompting)

# The AgentExecutor is the runtime for the agent.
# It invokes the agent, executes the chosen tools, and logs the process.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,  # Set to True to see the agent's thought process
    handle_parsing_errors=True,
)

def get_exif_data(file_path):
    img = Image.open(file_path)
    raw_exif_data = img.getexif()
    exif_data = {}
    exif_tags = {ExifTags.TAGS.get(tag, tag): value for tag, value in raw_exif_data.items()}
    for ifd_id in ExifTags.IFD:
        try:
            ifd = raw_exif_data.get_ifd(ifd_id)

            if ifd_id == ExifTags.IFD.GPSInfo:
                resolve = ExifTags.GPSTAGS
            else:
                resolve = ExifTags.TAGS

            exif_subcategory_data = {}
            for k, v in ifd.items():
                tag = resolve.get(k, k)
                exif_subcategory_data[tag] = v
            exif_data[ifd_id.name] = exif_subcategory_data
        except KeyError:
            pass
    return exif_data

def get_datetime(exif_data):
    if 'Exif' not in exif_data or 'DateTimeOriginal' not in exif_data['Exif']:
        return None

    date_time_original = exif_data['Exif']['DateTimeOriginal']
    offset_time_original = exif_data['Exif']['OffsetTimeOriginal']
    date_string = date_time_original + offset_time_original
    format_string = "%Y:%m:%d %H:%M:%S%z"
    dt_object = datetime.strptime(date_string, format_string)
    return dt_object

def coord_tuple_to_decimal(coord_tuple):
    return coord_tuple[0] + coord_tuple[1] / 60.0 + coord_tuple[2] / 3600.0

def coord_decimal_to_tuple(decimal_degrees):
    abs_decimal_degrees = abs(decimal_degrees)

    degrees = int(abs_decimal_degrees)
    minutes_float = (abs_decimal_degrees - degrees) * 60
    minutes = int(minutes_float)
    seconds_float = (minutes_float - minutes) * 60
    return (degrees, minutes, seconds_float)

def get_top_result(gmaps_reverse_geocode_results):
    location_type_priority = {
        'point_of_interest': 100000,
        'park': 100000,
        'airport': 100000,
        'natural_feature': 100000,
        'establishment': 90000,
        'premise': 90000,
        'subpremise': 90001,
        'street_address': 89999,
        'locality': 80000,
        # TODO: Add other location types.
        'default': 0
    }
    places_to_priority = []
    for result in gmaps_reverse_geocode_results:
        highest_priority_type = None
        for t in result['types']:
            if t in location_type_priority and (highest_priority_type is None or location_type_priority[highest_priority_type] < location_type_priority[t]):
                highest_priority_type = t
            elif t not in location_type_priority and highest_priority_type is None:
                highest_priority_type = 'default'
        places_to_priority.append((result['place_id'], location_type_priority[highest_priority_type]))
    places_to_priority.sort(key=lambda x: x[1], reverse=True)
    print(places_to_priority)
    return places_to_priority[0][0]

gmaps = googlemaps.Client()
def get_location(exif_data):
    if 'GPSInfo' not in exif_data:
        return None

    gps_info = exif_data['GPSInfo']
    if 'GPSLatitude' not in gps_info or 'GPSLongitude' not in gps_info or 'GPSLatitudeRef' not in gps_info or 'GPSLongitudeRef' not in gps_info:
        return None

    latitude = (1 if gps_info['GPSLatitudeRef'] == 'N' else -1) * coord_tuple_to_decimal(gps_info['GPSLatitude'])
    longitude = (1 if gps_info['GPSLongitudeRef'] == 'E' else -1) * coord_tuple_to_decimal(gps_info['GPSLongitude'])

    reverse_geocode_results = gmaps.reverse_geocode((latitude, longitude))
    if not reverse_geocode_results:
        return None
    top_result_place_id = get_top_result(reverse_geocode_results)

    place_details = gmaps.place(top_result_place_id)

    return place_details['result']['name']

gmaps = googlemaps.Client()
def get_lat_lng(place_name):
    geocode_results = gmaps.geocode(place_name)
    
    # Check if any results were returned
    if geocode_results:
        # The first result is usually the most relevant.
        # Extract latitude and longitude from the 'location' field of the geometry
        location = geocode_results[0]["geometry"]["location"]
        latitude = location["lat"]
        longitude = location["lng"]
        return (latitude, longitude)
    else:
        print(f"No results found for place name: '{place_name}'")
        return None

def lookup_exif_tag(tag_name):
    for tag, name in ExifTags.TAGS.items():
        if name == tag_name:
            return tag
    return None

def lookup_gps_exif_tag(tag_name):
    for tag, name in ExifTags.GPSTAGS.items():
        if name == tag_name:
            return tag
    return None

def set_coordinates_exif(filepath, lat, lng):
    lat_coord = coord_decimal_to_tuple(lat)
    lng_coord = coord_decimal_to_tuple(lng)
    print(lat_coord, lng_coord)
    image = Image.open(filepath)
    exif_dict = image.getexif() or {}
    if lookup_exif_tag('GPSInfo') not in exif_dict:
        exif_dict[lookup_exif_tag('GPSInfo')] = {}
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag('GPSLatitude')] = lat_coord
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag('GPSLatitudeRef')] = 'N' if lat >= 0 else 'S'
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag('GPSLongitude')] = lng_coord
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag('GPSLongitudeRef')] = 'E' if lng >= 0 else 'W'
    
    print(exif_dict)
    image.save(filepath, exif=exif_dict)



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
        print(f"Error: '{directory_path}' is not a valid directory or does not exist.")
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
                exif_data = get_exif_data(file_path)
                image_info_list.append({
                    'filepath': file_path,
                    'datetime': get_datetime(exif_data),
                    'location': get_location(exif_data),
                })
        # Do not go into subfolders
        break
    
    return image_info_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asdf todo."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="The path to the directory of images."
    )

    args = parser.parse_args()
    
    # Get the absolute path to ensure consistency
    target_directory = os.path.abspath(args.filepath)

    print(f"Scanning '{target_directory}' for image files...")
    images = find_image_files(target_directory)
    images.sort(key=lambda x: x['datetime'])
    for image in images:
        image['datetime'] = image['datetime'].strftime("%Y-%m-%d %H:%M:%S%z")

    print(images)

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

    user_input_commit = input('These are the proposed location assignments for the images. Do you want to commit them? Y/n').strip()
    print(user_input_commit)

    if user_input_commit != 'Y':
        print('Not commiting changes. Terminating program.')
        exit()

    # Commit changes to EXIF.
    for filepath, changed_location in changed_locations.items():
        print(filepath, changed_location)
        lat, lng = get_lat_lng(changed_location)
        set_coordinates_exif(filepath, lat, lng)



