# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set the working directory
WORKDIR /request_classifier

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# copy requirements file firstly, so it saves time if rebuilt.
COPY ./requirements.txt .
# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Set the entry point to run the FastAPI app with Uvicorn
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
