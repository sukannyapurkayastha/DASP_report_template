# Use NVIDIA's CUDA base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set the working directory
WORKDIR /attitudes_classifier

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY requirements.txt ./
COPY main.py ./ 
COPY utils.py ./ 
COPY models ./models
COPY df_sentences.csv ./df_sentences.csv
COPY sentences_author.csv ./sentences_author.csv
COPY attitudes_desc.csv ./attitudes_desc.csv
COPY __init__.py ./

# Install Python dependencies and ensure GPU support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py"]
