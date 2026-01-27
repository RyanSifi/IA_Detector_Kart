import os
from roboflow import Roboflow

# It's recommended to set the API key as an environment variable
# from dotenv import load_dotenv
# load_dotenv()

# Roboflow API key
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("Roboflow API Key not found.")
    print("Please get your API key from https://universe.roboflow.com/settings/api")
    api_key = input("Enter your Roboflow API Key: ")
    # TODO: Offer to save the API key as an environment variable for future use.

try:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("go-kart-detection-rxgso")
    dataset = project.version(2).download("yolov8")
except Exception as e:
    print(f"Failed to download dataset: {e}")
    print("Please ensure your API key is correct and has access to the dataset.")
