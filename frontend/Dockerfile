FROM python:3.10-slim

# Set the working directory
WORKDIR /frontend

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy requirements file firstly, so it saves time if rebuilt.
COPY ./requirements.txt .
# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

# Command to run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]

