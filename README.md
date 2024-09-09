# InvisibleCities Vehicle Tracking & Prediction

## Overview

The InvisibleCities Team addresses the Bengaluru Mobility Challenge by implementing a sophisticated vehicle tracking and predecting system. Solution processes video clips from various cameras to predict vehicle counts and analyze turning patterns using advanced object detection and tracking models.

## Project Features

1. **Vehicle Detection and Tracking**:
   - Utilizes YOLO models (`zm.pt` and `timestamp.pt`) for detecting various vehicle types and timestamp if available.
   - Tracks vehicles across frames using BoT-SORT, ensuring unique IDs for continuous tracking.

2. **Turning Patterns and Vehicle Counts**:
   - Analyzes vehicle movement to detect turning patterns.
   - Tracks vehicle counts in predefined regions and generates predictions for future vehicle counts.

3. **Output and Reporting**:
   - Generates JSON files with detailed vehicle counts, turning patterns, and tracking information and predicted counts


4. **Hardware and Computational Requirements (10%)**:
   - Tested on: Core i9 CPU, RTX 4090 GPU, 64GB RAM


## Docker Image

### Building the Docker Image

To build the Docker image, use the following Dockerfile configuration:

```dockerfile
# Use an appropriate base image with Python 3.12 and CUDA support for amd64 architecture
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies for OpenCV and other libraries
RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common build-essential libffi-dev libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev \
    libbz2-dev libexpat1-dev liblzma-dev libffi-dev libssl-dev libgl1-mesa-glx libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install -y python3.12 python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 312 && \
    update-alternatives --config python3 && \
    apt install python3.12-distutils -y && \
    apt install wget -y && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    apt-get install -y git

# Set the working directory inside the container
WORKDIR /app
ENV OPENAI_API_KEY="OCR_KEY"

# Copy your application files into the container
COPY . /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Set the entry point for the container
# ENTRYPOINT ["python3.12", "app.py"]

# Default command to pass to the entry point
CMD ["ENTRYPOINT", "<AppName>", "<IP_JSON>", "<OP_JSON>"]
```
# Building and Running the Docker Image

## BUILD
 
```sh
docker build --platform linux/amd64 -t zesstahyderabad/bmc-phase1:predict .
```

## RUN

```
docker run --rm --runtime=nvidia --gpus all -v <host-files-path>:/app/data
<image-name>:<image-tag> python3 app.py /app/data/input_file.json /app/data/output_file.json
```
## Example

```sh
docker run --rm --runtime=nvidia --gpus all -v /Zessta/Volumes/VID_STORE/:/app/data
zesstahyderabad/bmc-phase1:predict python3 app.py /app/data/input_file.json /app/data/output_file.json

```

# Names and descriptions

### Detector.py 

processes video with the model ```zm.pt``` and save the track count and returns a ``` dict ``` with 
```json
{
    "turning_pattern_N": {
        "vehicle_type_N":Count
    }
}
```

### Predictor.py 

Predictor uses calculated scales to check the count and predict the future counts by current passed vehchile counts and return value based on the time_scale


### TimeTracker.py 

Uses `timestamp.pt` and detect time_stamp frame and crops then saves to process image for getting timestamp and returns a `string->HH:MM:SS`

### app.py

Main exec file and takes input as `JSON` and processes it with proper key matching ID for JSON files stored in the JSONs folders and passed the PARAMETERS to detect_track with video_path, regions and turning_logics

then process the return `JSON` and calculates the PRED_COUNT and saves `JSON` in desired `PATH` from the Terminal





