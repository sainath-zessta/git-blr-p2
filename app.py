import json
import os
import numpy as np
import random
from PIL import Image, ImageDraw
import sys
from DetectionYolo.ProcessFrame import process_video
from DetectionYolo.EffNet import vehicle_reid_process
# Vehicle classes to generate matrices and images for
vehicle_classes = ['Car', 'Bus', 'Truck', 'Three-Wheeler', 'Two-Wheeler', 'LCV', 'Bicycle']

# Load the input JSON to get camera details
def load_input_file(input_json_path):
    with open(input_json_path, 'r') as f:
        camera_data = json.load(f)
    return camera_data

# Create directories for vehicle classes and generate images
def generate_images_and_matrices(camera_data, output_dir="/app/data/InvisibleCities"):
    N = len(camera_data)  # N refers to the number of cameras

    for camera_id,camera_path_list in camera_data.items():
        for video in camera_path_list:
            process_video(video, f"/app/cropped/{camera_id}")

    vehicle_reid_process("/app/cropped/",output_dir )




    print("Matrices and images created successfully.")

# Main function to run the entire process
def main(input_json_path):
    # Load camera data from the input file
    camera_data = load_input_file(input_json_path)

    # Generate images and matrices based on the input JSON
    generate_images_and_matrices(camera_data,op_folder)

if __name__ == "__main__":
    # The input file should be passed as an argument
    print("App Running " + sys.argv[0])
    ip_json = sys.argv[1]
    op_folder=sys.argv[2]
    main(ip_json,op_folder)
