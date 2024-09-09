# Base image with Python and Nvidia GPU support
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip python3-dev

# Copy the current directory contents into the container at /app
WORKDIR /app

# Copy files needed for execution
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Command to run the application


# Default command to pass to the entry point
CMD ["ENTRYPOINT", "<AppName>", "<IP_JSON>", "OP_DIR"]