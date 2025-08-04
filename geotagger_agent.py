import base64
import os

import json
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import mimetypes
from typing import List, Dict, Tuple, Any


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.0, api_key=os.getenv("GEMINI_API_KEY"))


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
        model="gemini-2.5-flash", temperature=0.0, api_key=os.getenv("GEMINI_API_KEY"))

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


def get_geotagger_agent():
    tools = [
        Tool(
            name="AnalyzeLocationsByTime",
            func=analyze_locations_by_time,
            description="Useful for when you want to determine which photos are likely to be taken at the same location as other photos, purely based on the time they were taken at. You should input a list of images. Each image should be a dictionary with the filepath, location, and datetime. Images where we don't know the location will have a null location value.",
        ),
        Tool(
            name="AnalyzeLocationsByPhoto",
            func=analyze_locations_by_photo,
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

    return agent_executor
